from ResNet_build_sample import ResnetBuilder
from keras.utils import np_utils
from keras.datasets import cifar10

nb_classes = 10


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)


input_shape = (32,32, 3)  # モデルの入力サイズ
num_classes = 10  # クラス数


# モデルを作成する。
model = ResnetBuilder.build_resnet_18(input_shape, num_classes)

model.compile(optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

model.fit(x_train, y_train,
        batch_size=256,
        epochs=200,
        verbose=1)

score = model.evaluate(x_test, y_test)
print(score)

# モデルをプロットする。
#from keras.utils import plot_model
#plot_model(model, to_file='resnet-model.png', 
#           show_shapes=True, show_layer_names=True)