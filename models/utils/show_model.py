from torchsummary import summary
from torchview import draw_graph

def show_model(model, Modelname="no", input_size=(3, 256, 256), depth=5):
    print("在这里打印Model********************************************************************************************************")
    summary(model, input_size=input_size)
    model_graph = draw_graph(model, input_size=(32,) + input_size, expand_nested=True,
                             hide_module_functions=True, hide_inner_tensors=True, roll=False,
                             filename=Modelname, directory="modelpng", depth=depth)

    model_graph.visual_graph.render(filename=Modelname, format='svg')
    model_graph.visual_graph.render(filename=Modelname, format='png')