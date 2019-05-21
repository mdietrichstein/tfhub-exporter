# Tensorflow Hub Module Exporter

This tool allows you to create optimized versions of the pretrained machine learning modules available on [Tensorflow Hub](https://tfhub.dev).

It downloads a module from TF Hub, freezes the network weights, applies a (configurable) number of transformations in order to optimize it for deployment and inference and exports the processed model as a [Saved Model](https://www.tensorflow.org/guide/saved_model).

## Prerequisites

* Make sure that you are running at least `python 3.6`
* Run `pip install -r requirements.txt` to install the neccessary dependencies

That's it, you should be ready to go.

## Usage

The examples in this repository use the [TF Hub Imagenet (ILSVRC-2012-CLS) classification with MobileNet V2 (depth multiplier 1.40)](https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/3) model.

Take a look at the [usage notebook](usage.ipynb) for a full example on how to export and load an optimized model.

### Export a TF Hub Module (export)

**Usage**

```
python main.py export --help

Usage: main.py export [OPTIONS] MODULE_URL TARGET

  Exports a frozen and optimized TF Hub module to a SavedModel

  Example: python main.py export "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/3"
  export/mobilenet

Options:
  --verbose          Enable verbose log output
  --signature TEXT   Model signature to export
  --transforms TEXT  Transforms which should be applied to the exported graph  [default: remove_nodes(op=Identity),
                     merge_duplicate_nodes, strip_unused_nodes, fold_constants(ignore_errors=true), fold_batch_norms]
  --help             Show this message and exit.
```

**Example**

```
python main.py export "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/3" export/mobilenet


EXPORT SUMMARY

Export Location: "/Users/marc/Development/tfhub-exporter/export/mobilenet"

 Model Inputs
╭────────┬─────────────────────┬─────────╮
│  name  │        shape        │  dtype  │
├────────┼─────────────────────┼─────────┤
│ images │ (None, 224, 224, 3) │ float32 │
╰────────┴─────────────────────┴─────────╯

 Model Outputs for signature "default"
╭──────────────────────────────────────────────────┬──────────────┬─────────╮
│                       name                       │    shape     │  dtype  │
├──────────────────────────────────────────────────┼──────────────┼─────────┤
│ module_apply_default/MobilenetV2/Logits/output:0 │ (None, 1001) │ float32 │
╰──────────────────────────────────────────────────┴──────────────┴─────────╯
```

### Inspect a TF Hub Module (show-info)

This command inspects the given module and displays information about the network inputs and outputs per signature.

**Usage**
```
python main.py show-info --help

Usage: main.py show-info [OPTIONS] MODULE_URL

  Shows internal TF Hub Module information like supported inputs, outputs and signatures

Options:
  --verbose  Enable verbose log output
  --help     Show this message and exit.
```

**Example**

```
python main.py show-info "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/3"

 Model Inputs
╭────────┬─────────────────────┬─────────╮
│  name  │        shape        │  dtype  │
├────────┼─────────────────────┼─────────┤
│ images │ (None, 224, 224, 3) │ float32 │
╰────────┴─────────────────────┴─────────╯

 Supported Outputs
╭───────────┬──────────────┬─────────╮
│ signature │    shape     │  dtype  │
├───────────┼──────────────┼─────────┤
│  default  │ (None, 1001) │ float32 │
╰───────────┴──────────────┴─────────╯
```