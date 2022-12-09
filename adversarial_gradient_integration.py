from typing import Optional

import tensorflow as tf


EPSILON_DEFAULT = 0.01
MAX_STEPS_DEFAULT = 100


def perturb_image(
        image: tf.Tensor,
        grad_adv: tf.Tensor,
        grad_lab: tf.Tensor,
        epsilon: float = EPSILON_DEFAULT,
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Perturb an image toward an adversarial example by epsilon.

    :param image: the image to perturbe
    :param grad_adv: the gradient with respect to the target label
    :param grad_lab: the gradient with respect to the initial prediction
    :param epsilon: the step size
    :return: the perturbed image and the gradient integration between the two images
    """
    grad_lab_norm = tf.norm(grad_lab, ord=2)
    delta = epsilon * tf.sign(grad_adv / grad_lab_norm)
    perturbed_image = image + delta
    perturbed_image_clipped = tf.clip_by_value(perturbed_image, 0, 255)
    delta = perturbed_image_clipped - image
    decrement = grad_lab*delta
    return perturbed_image_clipped, decrement


@tf.function
def find_adversarial_example(
        model: tf.keras.Model,
        init_image: tf.Tensor,
        init_label: tf.Tensor,
        target_label: tf.Tensor,
        epsilon: float = EPSILON_DEFAULT,
        max_steps: float = MAX_STEPS_DEFAULT,
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Find an adversarial example of an image given a target class.

    :param model: the model to optimize
    :param init_image: the image to start with
    :param init_label: the label associated with this image
    :param target_label: the label of the target class
    :param epsilon: the step size
    :param max_steps: the maximum number of steps to find an adversarial example
    :return: the gradient integrated along the followed path and the adversarial example
    """
    perturbed_image = tf.identity(init_image)
    agi = tf.zeros_like(init_image)
    j = 0
    adversarial_example_found = tf.constant(False)
    while not adversarial_example_found and j < max_steps:
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(perturbed_image)
            outputs = model(perturbed_image)
            loss_t = outputs[0, target_label]
            loss_i = outputs[0, init_label]
        grads_t = tape.gradient(loss_t, perturbed_image)
        grads_i = tape.gradient(loss_i, perturbed_image)
        del tape
        perturbed_image, decrement = perturb_image(perturbed_image, grads_t, grads_i, epsilon)
        agi -= decrement
        j += 1
        adversarial_example_found = tf.argmax(outputs, axis=-1)[0] == target_label
    if not adversarial_example_found:
        tf.print("Max number of steps reached before finding an adversarial example. Increase max_steps or epsilon.")
    return agi, perturbed_image


@tf.function
def explain(
        model: tf.keras.Model,
        image: tf.Tensor,
        sample_k: Optional[int] = None,
        epsilon: float = EPSILON_DEFAULT,
        max_steps: int = MAX_STEPS_DEFAULT,
        parallel_iterations: int = 12
) -> tf.Tensor:
    """
    Explain an image using AGI.

    :param model: the model to use
    :param image: the image to explain (the first dimension must be 1, the last dimension is the channels)
    :param sample_k: the number of classes to use to aggregate the AGIs
    :param epsilon: the step size
    :param max_steps: the maximum number of iterations to find an adversarial example
    :param parallel_iterations: the number of features (pixel R, G or B) to process in parallel
    :return: the AGI contribution of each feature
    """
    if sample_k is None:
        sample_k = model.output.shape[-1]
    else:
        if sample_k > model.output.shape[-1]:
            raise ValueError('sample_k cannot be bigger than the number of classes')
    if image.shape[0] != 1:
        raise ValueError(f"The first dimension of the sample must be 1, got {image.shape}")
    init_pred = tf.argmax(model(image), axis=-1)[0]
    # pick k classes making sure that the initial prediction is not in the set
    target_classes = tf.random.shuffle(
        tf.concat(
            [tf.range(init_pred),
             tf.range(init_pred + 1, model.output_shape[-1])],
            axis=0
        )
    )[:sample_k]
    individual_agis, _ = tf.map_fn(
        lambda target: find_adversarial_example(
            model=model,
            init_image=image,
            init_label=init_pred,
            target_label=target,
            epsilon=epsilon,
            max_steps=max_steps
        ),
        target_classes,
        fn_output_signature=(
            tf.TensorSpec(shape=image.shape, dtype=tf.float32),
            tf.TensorSpec(shape=image.shape, dtype=tf.float32)
        ),
        parallel_iterations=parallel_iterations
    )
    agis = tf.reduce_sum(individual_agis, axis=(0, 1))
    return agis


@tf.function
def explain_batch(
        model: tf.keras.Model,
        images: list[tf.Tensor],
        sample_k: Optional[int] = None,
        epsilon: float = EPSILON_DEFAULT,
        max_steps: int = MAX_STEPS_DEFAULT,
        parallel_iterations: int = 12,
) -> tf.Tensor:
    """
    Explain a batch of images using AGI.

    :param model: the model to use
    :param images: the images to explain (the first dimension is the batch, the last dimension is the channels)
    :param sample_k: the number of classes to use to aggregate the AGIs
    :param epsilon: the step size
    :param max_steps: the maximum number of iterations to find an adversarial example
    :param parallel_iterations: the number of images to process in parallel (not sure how this works with nested map_fn)
    :return: the AGI contribution of each feature
    """
    agis = tf.map_fn(
        lambda image: explain(
            model=model,
            image=image,
            sample_k=sample_k,
            epsilon=epsilon,
            max_steps=max_steps,
            parallel_iterations=parallel_iterations,
        ),
        images[:, tf.newaxis, ...],
        parallel_iterations=parallel_iterations,
    )
    return agis
