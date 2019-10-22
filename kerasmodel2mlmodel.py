import coremltools
import os.path
from keras.models import load_model

# Your keras model name and it will be the mlmodel name
model_name = 'LR_model'

# Your class labels in mlmodel
class_labels = 'LR/class_labels.txt'

def convert_model(model):
    print('converting...')

    # Use CoreML tools to convert keras model to ml model
    coreml_model = coremltools.converters.keras.convert(model,
                                                        input_names=['data'],
                                                        image_input_names='data',
                                                        class_labels=class_labels,
                                                        is_bgr=True)
    # Author description
    coreml_model.author = 'Ryan'

    # Short description for this model
    coreml_model.short_description = 'Indoor Positioning with Direction'

    # Input name and description for this model
    coreml_model.input_description['data'] = 'Indoor Image'

    # Output name and description for this model
    coreml_model.output_description['output1'] = 'Prediction'

    # Save .mlmodel file
    coreml_model.save(model_name + '.mlmodel')

    print('model converted')


if os.path.isfile(model_name + '.h5'):
    model = load_model(model_name + '.h5')
    convert_model(model)
else:
    print('no model found')
