from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.applications.efficientnet as efficientnet

from adversarial_gradient_integration import explain_batch

preprocessor = efficientnet.preprocess_input
decoder = efficientnet.decode_predictions
model = efficientnet.EfficientNetB0()
# model.summary()

images_path = './images'
sample_k = 15
epsilon = .05


images_path = Path(images_path)

def decode_image(path: Path, shape: tuple[int, int]):
    image_uint8 = tf.image.decode_jpeg(tf.io.read_file(str(path)))
    image_float = tf.cast(image_uint8, tf.float32)
    image_resized = tf.image.resize(image_float, shape)
    image_preprocessed = preprocessor(image_resized)
    return image_preprocessed


paths = [img_path for img_path in images_path.iterdir() if img_path.suffix in ['.jpg', '.jpeg']]
print(paths)
images = tf.stack([decode_image(path, model.input.shape[1:-1]) for path in paths], axis=0)

for path, prediction in zip(paths, decoder(model.predict(images), top=3)):
    print(f'{path.stem}: {", ".join([f"{name}, ({proba:.3f})" for _, name, proba in prediction])}')

agis = explain_batch(
    model=model,
    images=images,
    sample_k=sample_k,
    epsilon=epsilon,
)

print(agis)

# reduce over channels
agis_reduced = tf.reduce_max(agis, axis=-1)

outpath = images_path.parent / 'agi_images'
outpath.mkdir(exist_ok=True)
for path, agi in zip(paths, agis_reduced):
    plt.imshow(agi, cmap='gray')
    plt.savefig(outpath / (f'{path.stem}_agi_k={sample_k}_eps={epsilon}.png'))
    plt.show()
