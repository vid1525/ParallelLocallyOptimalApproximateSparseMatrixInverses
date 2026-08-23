from methods_cython import global_spai, inner_outer


METHOD_FAMILY_GLOBAL_SPAI = 'global_spai'
METHOD_FAMILY_INNER_OUTER = 'inner_outer'
FIELD_TITLE = 'title'
FIELD_OUTPUT_SLUG = 'output_slug'
FIELD_CYTHON_METHODS_HOLDER = 'cython_methods_holder'
FIELD_AVAILABLE_METHODS = 'available_methods'


METHOD_FAMILIES = {
    METHOD_FAMILY_GLOBAL_SPAI: {
        FIELD_TITLE: 'Global SPAI',
        FIELD_OUTPUT_SLUG: 'global_spai',
        FIELD_CYTHON_METHODS_HOLDER: global_spai,
    },
    METHOD_FAMILY_INNER_OUTER: {
        FIELD_TITLE: 'Inner-outer',
        FIELD_OUTPUT_SLUG: 'inner_outer',
        FIELD_CYTHON_METHODS_HOLDER: inner_outer,
    },
}


def get_method_label(family: dict[str, object], method_name: str) -> str:
    return f'{family[FIELD_TITLE]} {method_name.upper()}'


def get_plot_style(family: dict[str, object], method_name: str) -> dict[str, str]:
    """Use the paper's family colors and distinguish methods by line style."""
    line_styles = {
        'cg': '-',
        'minres': '-',
        'mr': '--',
        'lomr': ':',
    }
    return {
        'color': 'black' if family[FIELD_OUTPUT_SLUG] == METHOD_FAMILY_GLOBAL_SPAI else 'red',
        'linestyle': line_styles[method_name],
    }


def get_preconditioner_result_folder(family: dict[str, object], method_name: str) -> str:
    return f'{family[FIELD_OUTPUT_SLUG]}_{method_name}'


def get_method_families(name: str) -> list[dict[str, object]]:
    if name == 'all':
        return list(METHOD_FAMILIES.values())

    if name in METHOD_FAMILIES:
        return [METHOD_FAMILIES[name]]
    
    raise ValueError(f'options: all, {", ".join(METHOD_FAMILIES.keys())}')


def generate_methods_by_family(family: dict[str, object]):
    methods_holder = family[FIELD_CYTHON_METHODS_HOLDER]
    for method_name in getattr(methods_holder, FIELD_AVAILABLE_METHODS, ()):
        method = getattr(methods_holder, method_name)
        if callable(method):
            yield method_name, method
