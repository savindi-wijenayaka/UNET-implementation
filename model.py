from tensorflow.keras.layers import Input, Conv2D, Dropout, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

class UNET:
  def __init__(self, img_width, img_hieght, img_channels, num_filters, num_classes, activation):
      self.img_width = img_width
      self.img_hieght = img_hieght
      self.img_channels = img_channels
      self.num_filters = num_filters
      self.num_classes = num_classes
      self.activation = activation

  def __get_double_convos(self, step_input, num_filter, step_name, dropout=0.2):
    step_output = Conv2D(num_filter, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', name=step_name+'_layer1')(step_input)
    step_output = Dropout(dropout, name=step_name+'_dropout')(step_output)
    skip_connecton = Conv2D(num_filter, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', name=step_name+'_layer2')(step_output)
    return skip_connecton


  def get_model(self):
    skip_connections = []
    inputs = Input((self.img_width, self.img_hieght, self.img_channels), name='input_layer')
    step_input = inputs

    # Contraction path
    for index, num_filter in enumerate(self.num_filters):
      step_name = 'down_step'+str(index)
      skip_connecton = self.__get_double_convos(step_input, num_filter, step_name)
      step_input = MaxPooling2D((2, 2), name=step_name+'_maxpool')(skip_connecton)
      skip_connections.append(skip_connecton)

    # Bottom step
    bottom_num_filter = self.num_filters[-1]*2
    bottom_step_output = self.__get_double_convos(step_input, bottom_num_filter, 'bottomstep')

    skip_connections.reverse()
    step_input = bottom_step_output

    # Expansive path
    for index, num_filter in enumerate(reversed(self.num_filters)):
      step_name = 'up_step'+str(index)
      up_sample = Conv2DTranspose(num_filter, (2, 2), strides=(2, 2), padding='same', name=step_name+'upsample')(step_input)
      step_input = concatenate([up_sample, skip_connections[index]], name=step_name+'concatinate')
      step_input = self.__get_double_convos(step_input, num_filter, step_name)

    outputs = Conv2D(self.num_classes, (1, 1), activation=self.activation, name='final_output')(step_input)

    model = Model(inputs=[inputs], outputs=[outputs], name='UNET_from_scratch')
    return model
