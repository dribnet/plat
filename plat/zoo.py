import os
import sys
import importlib
import tempfile
import gzip
import shutil
from distutils.util import strtobool
from fuel.downloaders.base import default_downloader

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

def helpful_interface_message_exit(model_interface, e):
    print("==> Failed to load supporting class {}".format(model_interface))
    print("==> Check that package {} is installed".format(model_interface.split(".")[0]))
    print("(exception was: {})".format(e))
    sys.exit(1)

def load_model_with_interface(model_file_name, model_interface):
    model_class_parts = model_interface.split(".")
    model_class_name = model_class_parts[-1]
    model_module_name = ".".join(model_class_parts[:-1])
    print("Loading {} interface from {}".format(model_class_name, model_module_name))        
    try:
        ModelClass = getattr(importlib.import_module(model_module_name), model_class_name)
    except ImportError as e:
        helpful_interface_message_exit(model_interface, e)
    print("Loading model {}".format(os.path.basename(model_file_name)))
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
            model_type = model_file_name.split(".")[-1]
        model_interface = model_interface_table[model_type]

    return load_model_with_interface(model_file_name, model_interface)

# this obviously won't scale, but is fine for now
model_download_table = {
    "celeba_64.discgen": "http://drib.net/platzoo/celeba_64_v1.0.0.discgen.gz"
}

def download_model(model_name):
    # see if this is a known model
    if not model_name in model_download_table.keys():
        print("Failure: unknown model {}".format(model_name))
        sys.exit(1)

    # resolve url
    model_url = model_download_table[model_name]
    platzoo_dir = get_platzoo_dir()
    local_gz_filename = model_url.split("/")[-1]
    temp_dir = tempfile.mkdtemp()

    # download
    default_downloader(temp_dir, [model_url], [local_gz_filename])

    if local_gz_filename.endswith(".gz"):
        local_filename = local_gz_filename[:-3]
    else:
        local_filename = "{}.2".format(local_gz_filename)
    # convert to absolute paths
    final_local_filepath = os.path.join(platzoo_dir, local_filename)
    final_local_linkpath = os.path.join(platzoo_dir, model_name)
    temp_gz_filepath = os.path.join(temp_dir, local_gz_filename)
    temp_filepath = os.path.join(temp_dir, local_filename)

    # decompress the file to temporary location
    print("Decompressing {}".format(model_name))
    with open(temp_filepath, 'wb') as f_out, gzip.open(temp_gz_filepath, 'rb') as f_in:
        shutil.copyfileobj(f_in, f_out)

    # atomic rename (prevents half-downloaded files)
    print("Installing {}".format(model_name))
    os.rename(temp_filepath, final_local_filepath)
    # symlink, removing old first if necessary
    if os.path.exists(final_local_linkpath):
        os.remove(final_local_linkpath)
    os.symlink(final_local_filepath, final_local_linkpath)

    # cleanup temp directory
    # TODO: try/catch the download for failure cleanup
    shutil.rmtree(temp_dir)

# check if model exists. if not, but can be downloaded, prompt for download
def check_model_download(model_name):
    filename = resolve_model_to_filename(model_name)
    if os.path.exists(filename):
        return
    if not model_name in model_download_table.keys():
        print("Failure: unknown model {}".format(model_name))
        sys.exit(1)
    while True:
        try:
            result = raw_input("Model {} will be downloaded. Continue? [Y/n]".format(model_name))
            if result == "" or strtobool(result):
                download_model(model_name)
                return
            else:
                print("No model, cannot coninue")
        except ValueError:
            pass
