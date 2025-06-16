from keras.utils import plot_model, model_to_dot
from keras.models import load_model
import visualkeras
from PIL import ImageFont

# 加载模型
model = load_model('regressionNew/cnn_lstm_transformer_model_optimized.h5')

# 方法 1：使用 plot_model
plot_model(
    model,
    to_file='model_architecture.png',
    show_shapes=True,
    show_layer_names=True,
    expand_nested=True,
    dpi=300,
    rankdir='TB',
)

# 方法 2：使用 visualkeras
font = ImageFont.truetype("arial.ttf", 20)
visualkeras.layered_view(
    model,
    to_file='model_visualkeras.png',
    legend=True,
    draw_volume=False,
    scale_xy=20,
    spacing=40,
    font=font,
).show()

# 方法 3：使用 graphviz
dot = model_to_dot(
    model,
    show_shapes=True,
    show_layer_names=True,
    rankdir='TB',
    dpi=300,
    expand_nested=True,
)
dot.write_png('model_graphviz.png')
dot.write_svg('model_graphviz.svg')