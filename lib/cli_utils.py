from beautifultable import BeautifulTable


def __to_shape_string(shape):
    return '(' + (', '.join(list(map(lambda x: str(x.value), shape)))) + ')'


def print_outputs(tensors):
    table = BeautifulTable()
    table.column_headers = ["signature", "shape", "dtype"]

    for name, value in tensors.items():
        table.append_row(
            [name, __to_shape_string(value.get_shape()), value.dtype.name])

    table.set_style(BeautifulTable.STYLE_BOX_ROUNDED)
    table.sort('signature')
    print(table)


def print_tensors(tensors):
    table = BeautifulTable()
    table.column_headers = ["name", "shape", "dtype"]

    for name, value in tensors.items():
        table.append_row(
            [name, __to_shape_string(value.shape), value.dtype.name])

    table.set_style(BeautifulTable.STYLE_BOX_ROUNDED)
    table.sort('name')
    print(table)


def print_tensor(tensor):
    table = BeautifulTable()
    table.column_headers = ["name", "shape", "dtype"]

    table.append_row([tensor.name, __to_shape_string(
        tensor.shape), tensor.dtype.name])

    table.set_style(BeautifulTable.STYLE_BOX_ROUNDED)
    table.sort('name')
    print(table)
