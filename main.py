import logging
import tensorflow as tf
import os
import sys
import click
import time
import tempfile

from enum import Enum, unique

from lib import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.logging.set_verbosity(tf.logging.ERROR)
tf.enable_eager_execution()


logging.root.setLevel(logging.NOTSET)
logging.basicConfig(level=logging.INFO)


def log_duration(t0):
    logging.info(f'<Took {(time.time() - t0):.2f}s>')


@click.group(chain=False)
def cli():
    pass


@cli.command(context_settings=dict(max_content_width=120))
@click.argument("module_url")
@click.option("--verbose", default=False, is_flag=True, help="Enable verbose log output")
def show_info(module_url, verbose):
    """Shows internal TF Hub Module information like supported inputs, outputs and signatures

    Example: python main.py show-info "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/3"
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    graph = tf.Graph()

    logging.info(f'Loading module "{module_url}"')
    t0 = time.time()
    module_info = get_module_info(graph, module_url)
    log_duration(t0)

    inputs, output, output_info = \
        module_info['inputs'], module_info['output'], module_info['output_info']

    print()
    print(' Model Inputs')
    print_tensors(inputs)
    print()
    print(' Supported Outputs')
    print_outputs(output_info)
    print()


@cli.command(context_settings=dict(max_content_width=120))
@click.argument("module_url")
@click.option("--verbose", default=False, is_flag=True, help="Enable verbose log output")
@click.argument("target")
@click.option("--signature", default='default', help="Model signature to export")
@click.option("--transforms", default=DEFAULT_TRANSFORMS, multiple=True,
              help="Transforms which should be applied to the exported graph",
              show_default=True)
def export(module_url, target, signature, verbose, transforms):
    """Exports a frozen and optimized TF Hub module to a SavedModel

    Example: python main.py export "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/3" export/mobilenet
    """

    if os.path.exists(target):
        sys.exit(
            f'The output target directory at "{target}" already exists. Please remove it or choose a different output target for this export.')

    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    graph = tf.Graph()

    logging.info(f'Loading module "{module_url}"')
    t0 = time.time()
    module_info = get_module_info(graph, module_url, signature)
    log_duration(t0)

    inputs, output, output_info, init_op = \
        module_info['inputs'], module_info['output'], module_info['output_info'], module_info['init_op']

    logging.debug(f'Detected inputs "{inputs}"')
    logging.debug(f'Output "{output.name}": "{output_info}"')

    output_name = output.name
    normalized_output_name = output_name.split(':')[0]
    logging.debug(f'Normalized output name: "{normalized_output_name}"')

    with tf.Session(graph=graph) as sess:
        sess.run(init_op)

        with tempfile.TemporaryDirectory() as export_directory:
            target_directory = export_directory + '/' + target

            logging.info(f'Exporting TF Hub module to "{target_directory}"')
            t0 = time.time()
            tf.saved_model.simple_save(sess, target_directory, inputs=inputs, outputs={
                'output': output
            })
            log_duration(t0)

            with tempfile.NamedTemporaryFile() as t:
                logging.info(f'Freezing Graph')
                saved_model_to_frozen_graph(
                    target_directory, t.name, normalized_output_name)
                log_duration(t0)

                log_directory = export_directory + '/log'
                logging.info(
                    f'Applying transforms {transforms}.\nLogs at "{log_directory}"')
                optimize_graph(t.name, t.name, list(inputs.keys()), normalized_output_name,
                               logdir=log_directory)
                log_duration(t0)

                logging.info(f'Exporting SavedModel to "{target}"')
                convert_graph_def_to_saved_model(target, t.name, output_name)
                log_duration(t0)

                print('\n\nEXPORT SUMMARY')
                print(f'\nExport Location: "{os.path.abspath(target)}"\n')
                print(' Model Inputs')
                print_tensors(inputs)
                print()
                print(f' Model Outputs for signature "{signature}"')
                print_tensor(output)


cli()
