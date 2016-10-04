import os
import importlib

def get_platzoo_dir():
    platzoo_dir = os.environ.get('PLATZOO_DIR')
    if platzoo_dir is None:
        platdir = os.environ.get('PLAT_DIR')
        if platdir is None:
            platdir = os.path.expanduser(os.path.join('~', '.plat'))
        platzoo_dir = os.path.join(platdir, 'zoo')
    if not os.path.exists(platzoo_dir):
        os.makedirs(platzoo_dir)
    return platzoo_dir

def resolve_model_to_filename(filename):
    platzoo_dir = get_platzoo_dir()
    pathname = os.path.join(platzoo_dir, filename)
    return pathname

def resolve_model_type_from_filename(filename):
    filename_dot_parts = filename.split(".")
    model_type = filename_dot_parts[-1]

model_interface_table = {
    "discgen": "discgen.interface.DiscGenModel"
}

def load_model_with_interface(model_file_name, model_interface):
    model_class_parts = model_interface.split(".")
    model_class_name = model_class_parts[-1]
    model_module_name = ".".join(model_class_parts[:-1])
    print("Loading {} interface from {}".format(model_class_name, model_module_name))        
    ModelClass = getattr(importlib.import_module(model_module_name), model_class_name)
    print("Loading model from {}".format(model_file_name))
    model = ModelClass(filename=model_file_name)
    print("Model loaded.")
    return model    

def load_model(model=None, model_file_name=None, model_type=None, model_interface=None):
    """
    Wrapper for resolving and loading a model.
    If model_file_name is not given, it is resolved to a filename in platzoo from the model name.
    If model_interface is not given, it is resolved from the model_type.
    Finally, the model_type itself might be resolved from the model_file_name if needed.

    After resolution, the model is loaded and returned.
    """

    if model_file_name == None:
        model_file_name = resolve_model_to_filename(model)

    if model_interface == None:
        if model_type == None:
            model_type = model_interface_table[model_file_name]
        model_interface = resolve_model_interface_from_type(model_type)

    return load_model_with_interface(model_file_name, model_interface)
