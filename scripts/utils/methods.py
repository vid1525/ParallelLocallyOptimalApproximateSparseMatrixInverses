from functools import partial

from methods_cython import global_spai, inner_outer


METHOD_FAMILY_GLOBAL_SPAI = 'global_spai'
METHOD_FAMILY_INNER_OUTER = 'inner_outer'
METHOD_FAMILY_INNER_OUTER_5 = 'inner_outer_5'
FIELD_TITLE = 'title'
FIELD_OUTPUT_SLUG = 'output_slug'
FIELD_CYTHON_METHODS_HOLDER = 'cython_methods_holder'
FIELD_AVAILABLE_METHODS = 'available_methods'
FIELD_INNER_ITERATIONS = 'inner_iterations'
FIELD_COLOR = 'color'


METHOD_FAMILIES = {
    METHOD_FAMILY_GLOBAL_SPAI: {
        FIELD_TITLE: 'Global',
        FIELD_OUTPUT_SLUG: 'global_spai',
        FIELD_CYTHON_METHODS_HOLDER: global_spai,
        FIELD_COLOR: 'black',
    },
    METHOD_FAMILY_INNER_OUTER: {
        FIELD_TITLE: 'Inner-outer - 2 inner iter',
        FIELD_OUTPUT_SLUG: 'inner_outer',
        FIELD_CYTHON_METHODS_HOLDER: inner_outer,
        FIELD_INNER_ITERATIONS: 2,
        FIELD_COLOR: 'red',
    },
    METHOD_FAMILY_INNER_OUTER_5: {
        FIELD_TITLE: 'Inner-outer - 5 inner iter',
        FIELD_OUTPUT_SLUG: 'inner_outer_5',
        FIELD_CYTHON_METHODS_HOLDER: inner_outer,
        FIELD_INNER_ITERATIONS: 5,
        FIELD_COLOR: 'blue',
    },
}


def get_method_label(family: dict[str, object], method_name: str) -> str:
    display_name = ('PCG' if family[FIELD_OUTPUT_SLUG] == METHOD_FAMILY_GLOBAL_SPAI
                    and method_name == 'cg' else method_name.upper())
    return f'{family[FIELD_TITLE]} {display_name}'


def get_plot_style(family: dict[str, object], method_name: str) -> dict[str, str]:
    line_styles = {
        'cg': '-',
        'mr': '--',
        'lomr': ':',
    }
    markers = {
        METHOD_FAMILY_GLOBAL_SPAI: {'cg': 'o', 'mr': 's', 'lomr': '^'},
        METHOD_FAMILY_INNER_OUTER: {'mr': 'D', 'lomr': 'v'},
        METHOD_FAMILY_INNER_OUTER_5: {'mr': 'P', 'lomr': 'X'},
    }
    return {
        'color': family[FIELD_COLOR],
        'linestyle': line_styles[method_name],
        'marker': markers[family[FIELD_OUTPUT_SLUG]][method_name],
    }


def get_preconditioner_result_folder(family: dict[str, object], method_name: str) -> str:
    return f'{family[FIELD_OUTPUT_SLUG]}_{method_name}'


def get_method_families(name: str) -> list[dict[str, object]]:
    if name == 'all':
        return list(METHOD_FAMILIES.values())

    if name == METHOD_FAMILY_INNER_OUTER:
        return [METHOD_FAMILIES[METHOD_FAMILY_INNER_OUTER], METHOD_FAMILIES[METHOD_FAMILY_INNER_OUTER_5]]

    if name in METHOD_FAMILIES:
        return [METHOD_FAMILIES[name]]
    
    raise ValueError(f'options: all, {", ".join(METHOD_FAMILIES.keys())}')


def generate_methods_by_family(family: dict[str, object]):
    methods_holder = family[FIELD_CYTHON_METHODS_HOLDER]
    for method_name in getattr(methods_holder, FIELD_AVAILABLE_METHODS, ()):
        method = getattr(methods_holder, method_name)
        if callable(method):
            if FIELD_INNER_ITERATIONS in family:
                method = partial(method, inner_iterations=family[FIELD_INNER_ITERATIONS])
            yield method_name, method
