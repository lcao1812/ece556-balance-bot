from keras import Layer
from keras.api import models
import visualkeras

# adapted from https://github.com/paulgavrikov/visualkeras/?tab=readme-ov-file#:~:text=The%20following%20function%20aims%20to%20describe%20the%20names%20of%20layers%20and%20their%20dimensionality.%20It%20would%20produce%20the%20output%20shown%20in%20the%20figure%20below
# This function is called for each layer in the model to get the text to display
# in the visualization. It returns the text and a boolean indicating whether
# the text should be drawn above or below the layer.
# The text is formatted to show the output shape of the layer and its name.


def text_callable(layer_index: int, layer: Layer):
    # Every other piece of text is drawn above the layer, the first one below
    above = bool(layer_index % 2)

    # Get the output shape of the layer
    output_shape = [x for x in list(layer.output.shape) if x is not None]

    # Variable to store text which will be drawn
    output_shape_txt = ""

    # Create a string representation of the output shape
    for ii in range(len(output_shape)):
        output_shape_txt += str(output_shape[ii])
        if ii < len(output_shape) - 2:  # Add an x between dimensions, e.g. 3x3
            output_shape_txt += "x"
        # Add a newline between the last two dimensions, e.g. 3x3 \n 64
        if ii == len(output_shape) - 2:
            output_shape_txt += "\n"

    # Add the name of the layer to the text, as a new line
    output_shape_txt += f"\n{layer.name}"

    # Return the text value and if it should be drawn above the layer
    return output_shape_txt, above


model = models.load_model('cnn.keras', compile=False)

# display using your system viewer
visualkeras.layered_view(
    model, legend=True, text_callable=text_callable, scale_xy=1, scale_z=1, max_z=1000, to_file='../Reports/cnn_visual.png')