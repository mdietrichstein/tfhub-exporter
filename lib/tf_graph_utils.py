import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.graph_util as graph_util
from tensorflow.saved_model import tag_constants
from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow.python import ops
from tensorflow.python.tools import freeze_graph

DEFAULT_TRANSFORMS = [
    'remove_nodes(op=Identity)',
    'merge_duplicate_nodes',
    'strip_unused_nodes',
    'fold_constants(ignore_errors=true)',
    'fold_batch_norms',
    #  'quantize_nodes',
    #  'quantize_weights',
]


def get_module_info(g, url, signature='default'):
    with g.as_default():
        module = hub.Module(url)

        inputs = {name: tf.placeholder(definition.dtype, definition.get_shape(), name)
                  for name, definition in module.get_input_info_dict().items()}

        output_info = module.get_output_info_dict(signature)

        output = module(inputs, signature=signature)
        output_name = output.name

        init_op = tf.group(
            [tf.global_variables_initializer(), tf.tables_initializer()])

    return {
        'inputs': inputs,
        'output': output,
        'output_info': output_info,
        'init_op': init_op
    }


def saved_model_to_frozen_graph(saved_model_dir, output_graph_path, output_node_name):
    freeze_graph.freeze_graph(
        input_saved_model_dir=saved_model_dir,
        output_graph=output_graph_path,
        saved_model_tags=tag_constants.SERVING,
        output_node_names=output_node_name,
        initializer_nodes='',
        input_graph=None,
        input_saver=False,
        input_binary=False,
        input_checkpoint=None,
        restore_op_name=None,
        filename_tensor_name=None,
        clear_devices=False,
        input_meta_graph=False)


def optimize_graph(input_graph_path, output_graph_path,
                   input_node_names, output_node_name,
                   transforms=DEFAULT_TRANSFORMS,
                   logdir='./log'):

    optimized_graph_def = TransformGraph(
        __get_graph_def_from_file(input_graph_path),
        input_node_names,
        [output_node_name], transforms)

    tf.train.write_graph(optimized_graph_def, logdir=logdir,
                         as_text=False, name=output_graph_path)


def convert_graph_def_to_saved_model(export_dir, graph_filepath, output_name):
    graph_def = __get_graph_def_from_file(graph_filepath)

    with tf.Session(graph=tf.Graph()) as session:
        tf.import_graph_def(graph_def, name='')
        tf.saved_model.simple_save(
            session,
            export_dir,
            inputs={
                node.name: session.graph.get_tensor_by_name(
                    '{}:0'.format(node.name))
                for node in graph_def.node if node.op == 'Placeholder'},

            outputs={'output': session.graph.get_tensor_by_name(output_name)}
        )


def __get_graph_def_from_file(graph_filepath):
    with ops.Graph().as_default():
        with tf.gfile.GFile(graph_filepath, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            return graph_def
