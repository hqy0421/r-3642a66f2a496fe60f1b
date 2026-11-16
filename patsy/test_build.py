# This file is part of Patsy
# Copyright (C) 2012-2013 Nathaniel Smith <njs@pobox.com>
# See file LICENSE.txt for license information.

# There are a number of unit tests in build.py, but this file contains more
# thorough tests of the overall design matrix building system. (These are
# still not exhaustive end-to-end tests, though -- for that see
# test_highlevel.py.)

import numpy as np
import pytest
from patsy import PatsyError
from patsy.util import atleast_2d_column_default, have_pandas, have_pandas_categorical
from patsy.desc import Term, INTERCEPT
from patsy.build import build_design_matrices, design_matrix_builders
from patsy.categorical import C
from patsy.user_util import balanced, LookupFactor
from patsy.design_info import DesignMatrix, DesignInfo

if have_pandas:
    import pandas


def assert_full_rank(m):
    m = atleast_2d_column_default(m)
    if m.shape[1] == 0:
        return True
    u, s, v = np.linalg.svd(m)
    rank = np.sum(s > 1e-10)
    assert rank == m.shape[1]


def test_assert_full_rank():
    assert_full_rank(np.eye(10))
    assert_full_rank([[1, 0], [1, 0], [1, 0], [1, 1]])
    pytest.raises(AssertionError, assert_full_rank, [[1, 0], [2, 0]])
    pytest.raises(AssertionError, assert_full_rank, [[1, 2], [2, 4]])
    pytest.raises(AssertionError, assert_full_rank, [[1, 2, 3], [1, 10, 100]])
    # col1 + col2 = col3
    pytest.raises(AssertionError, assert_full_rank, [[1, 2, 3], [1, 5, 6], [1, 6, 7]])


def make_termlist(*entries):
    terms = []
    for entry in entries:
        terms.append(Term([LookupFactor(name) for name in entry]))
    return terms


def check_design_matrix(mm, expected_rank, termlist, column_names=None):
    assert_full_rank(mm)
    assert set(mm.design_info.terms) == set(termlist)
    if column_names is not None:
        assert mm.design_info.column_names == column_names
    assert mm.ndim == 2
    assert mm.shape[1] == expected_rank


def make_matrix(data, expected_rank, entries, column_names=None):
    termlist = make_termlist(*entries)

    def iter_maker():
        yield data

    design_infos = design_matrix_builders([termlist], iter_maker, eval_env=0)
    matrices = build_design_matrices(design_infos, data)
    matrix = matrices[0]
    assert design_infos[0].term_slices == matrix.design_info.term_slices
    assert design_infos[0].column_names == matrix.design_info.column_names
    assert matrix.design_info is design_infos[0]
    check_design_matrix(matrix, expected_rank, termlist, column_names=column_names)
    return matrix


def test_simple():
    data = balanced(a=2, b=2)
    x1 = data["x1"] = np.linspace(0, 1, len(data["a"]))
    x2 = data["x2"] = data["x1"] ** 2

    m = make_matrix(data, 2, [["a"]], column_names=["a[a1]", "a[a2]"])
    assert np.allclose(m, [[1, 0], [1, 0], [0, 1], [0, 1]])

    m = make_matrix(data, 2, [[], ["a"]], column_names=["Intercept", "a[T.a2]"])
    assert np.allclose(m, [[1, 0], [1, 0], [1, 1], [1, 1]])

    m = make_matrix(
        data,
        4,
        [["a", "b"]],
        column_names=["a[a1]:b[b1]", "a[a2]:b[b1]", "a[a1]:b[b2]", "a[a2]:b[b2]"],
    )
    assert np.allclose(m, [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

    m = make_matrix(
        data,
        4,
        [[], ["a"], ["b"], ["a", "b"]],
        column_names=["Intercept", "a[T.a2]", "b[T.b2]", "a[T.a2]:b[T.b2]"],
    )
    assert np.allclose(m, [[1, 0, 0, 0], [1, 0, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]])

    m = make_matrix(
        data,
        4,
        [[], ["b"], ["a"], ["b", "a"]],
        column_names=["Intercept", "b[T.b2]", "a[T.a2]", "b[T.b2]:a[T.a2]"],
    )
    assert np.allclose(m, [[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]])

    m = make_matrix(
        data,
        4,
        [["a"], ["x1"], ["a", "x1"]],
        column_names=["a[a1]", "a[a2]", "x1", "a[T.a2]:x1"],
    )
    assert np.allclose(
        m,
        [
            [1, 0, x1[0], 0],
            [1, 0, x1[1], 0],
            [0, 1, x1[2], x1[2]],
            [0, 1, x1[3], x1[3]],
        ],
    )

    m = make_matrix(
        data, 3, [["x1"], ["x2"], ["x2", "x1"]], column_names=["x1", "x2", "x2:x1"]
    )
    assert np.allclose(m, np.column_stack((x1, x2, x1 * x2)))


def test_R_bugs():
    data = balanced(a=2, b=2, c=2)
    data["x"] = np.linspace(0, 1, len(data["a"]))
    # For "1 + a:b", R produces a design matrix with too many columns (5
    # instead of 4), because it can't tell that there is a redundancy between
    # the two terms.
    make_matrix(data, 4, [[], ["a", "b"]])
    # For "0 + a:x + a:b", R produces a design matrix with too few columns (4
    # instead of 6), because it thinks that there is a redundancy which
    # doesn't exist.
    make_matrix(data, 6, [["a", "x"], ["a", "b"]])
    # This can be compared with "0 + a:c + a:b", where the redundancy does
    # exist. Confusingly, adding another categorical factor increases the
    # baseline dimensionality to 8, and then the redundancy reduces it to 6
    # again, so the result is the same as before but for different reasons. (R
    # does get this one right, but we might as well test it.)
    make_matrix(data, 6, [["a", "c"], ["a", "b"]])


def test_redundancy_thoroughly():
    # To make sure there aren't any lurking bugs analogous to the ones that R
    # has (see above), we check that we get the correct matrix rank for every
    # possible combination of 2 categorical and 2 numerical factors.
    data = balanced(a=2, b=2, repeat=5)
    data["x1"] = np.linspace(0, 1, len(data["a"]))
    data["x2"] = data["x1"] ** 2

    def all_subsets(l):
        if not l:
            yield tuple()
        else:
            obj = l[0]
            for subset in all_subsets(l[1:]):
                yield tuple(sorted(subset))
                yield tuple(sorted((obj,) + subset))

    all_terms = list(all_subsets(("a", "b", "x1", "x2")))
    all_termlist_templates = list(all_subsets(all_terms))
    print(len(all_termlist_templates))
    # eliminate some of the symmetric versions to speed things up
    redundant = [
        [("b",), ("a",)],
        [("x2",), ("x1",)],
        [("b", "x2"), ("a", "x1")],
        [("a", "b", "x2"), ("a", "b", "x1")],
        [("b", "x1", "x2"), ("a", "x1", "x2")],
    ]
    count = 0
    import time

    start = time.time()
    for termlist_template in all_termlist_templates:
        termlist_set = set(termlist_template)
        for dispreferred, preferred in redundant:
            if dispreferred in termlist_set and preferred not in termlist_set:
                break
        else:
            expanded_terms = set()
            for term_template in termlist_template:
                numeric = tuple([t for t in term_template if t.startswith("x")])
                rest = [t for t in term_template if not t.startswith("x")]
                for subset_rest in all_subsets(rest):
                    expanded_terms.add(frozenset(subset_rest + numeric))
            # Because our categorical variables have 2 levels, each expanded
            # term corresponds to 1 unique dimension of variation
            expected_rank = len(expanded_terms)
            if termlist_template in [(), ((),)]:
                # No data dependence, should fail
                pytest.raises(
                    PatsyError, make_matrix, data, expected_rank, termlist_template
                )
            else:
                make_matrix(data, expected_rank, termlist_template)
            count += 1
            if count % 100 == 0:
                print("Completed:", count)
    print("Took %0.2f seconds" % (time.time() - start,))


test_redundancy_thoroughly.slow = 1


def test_data_types():
    basic_dict = {"a": ["a1", "a2", "a1", "a2"], "x": [1, 2, 3, 4]}
    # On Python 2, this is identical to basic_dict:
    basic_dict_bytes = dict(basic_dict)
    basic_dict_bytes["a"] = [s.encode("ascii") for s in basic_dict_bytes["a"]]
    # On Python 3, this is identical to basic_dict:
    basic_dict_unicode = {"a": ["a1", "a2", "a1", "a2"], "x": [1, 2, 3, 4]}
    basic_dict_unicode = dict(basic_dict)
    basic_dict_unicode["a"] = [str(s) for s in basic_dict_unicode["a"]]

    structured_array_bytes = np.array(
        list(zip(basic_dict["a"], basic_dict["x"])), dtype=[("a", "S2"), ("x", int)]
    )
    structured_array_unicode = np.array(
        list(zip(basic_dict["a"], basic_dict["x"])), dtype=[("a", "U2"), ("x", int)]
    )
    recarray_bytes = structured_array_bytes.view(np.recarray)
    recarray_unicode = structured_array_unicode.view(np.recarray)
    datas = [
        basic_dict,
        structured_array_bytes,
        structured_array_unicode,
        recarray_bytes,
        recarray_unicode,
    ]
    if have_pandas:
        df_bytes = pandas.DataFrame(basic_dict_bytes)
        datas.append(df_bytes)
        df_unicode = pandas.DataFrame(basic_dict_unicode)
        datas.append(df_unicode)
    for data in datas:
        m = make_matrix(
            data,
            4,
            [["a"], ["a", "x"]],
            column_names=["a[a1]", "a[a2]", "a[a1]:x", "a[a2]:x"],
        )
        assert np.allclose(m, [[1, 0, 1, 0], [0, 1, 0, 2], [1, 0, 3, 0], [0, 1, 0, 4]])


def test_build_design_matrices_dtype():
    data = {"x": [1, 2, 3]}

    def iter_maker():
        yield data

    builder = design_matrix_builders([make_termlist("x")], iter_maker, 0)[0]

    mat = build_design_matrices([builder], data)[0]
    assert mat.dtype == np.dtype(np.float64)

    mat = build_design_matrices([builder], data, dtype=np.float32)[0]
    assert mat.dtype == np.dtype(np.float32)

    if hasattr(np, "float128"):
        mat = build_design_matrices([builder], data, dtype=np.float128)[0]
        assert mat.dtype == np.dtype(np.float128)


def test_return_type():
    data = {"x": [1, 2, 3]}

    def iter_maker():
        yield data

    builder = design_matrix_builders([make_termlist("x")], iter_maker, 0)[0]

    # Check explicitly passing return_type="matrix" works
    mat = build_design_matrices([builder], data, return_type="matrix")[0]
    assert isinstance(mat, DesignMatrix)

    # Check that nonsense is detected
    pytest.raises(
        PatsyError, build_design_matrices, [builder], data, return_type="asdfsadf"
    )


def test_NA_action():
    initial_data = {"x": [1, 2, 3], "c": ["c1", "c2", "c1"]}

    def iter_maker():
        yield initial_data

    builder = design_matrix_builders([make_termlist("x", "c")], iter_maker, 0)[0]

    # By default drops rows containing either NaN or None
    mat = build_design_matrices(
        [builder],
        {"x": [10.0, np.nan, 20.0], "c": np.asarray(["c1", "c2", None], dtype=object)},
    )[0]
    assert mat.shape == (1, 3)
    assert np.array_equal(mat, [[1.0, 0.0, 10.0]])

    # NA_action="a string" also accepted:
    mat = build_design_matrices(
        [builder],
        {"x": [10.0, np.nan, 20.0], "c": np.asarray(["c1", "c2", None], dtype=object)},
        NA_action="drop",
    )[0]
    assert mat.shape == (1, 3)
    assert np.array_equal(mat, [[1.0, 0.0, 10.0]])

    # And objects
    from patsy.missing import NAAction

    # allows NaN's to pass through
    NA_action = NAAction(NA_types=[])
    mat = build_design_matrices(
        [builder],
        {"x": [10.0, np.nan], "c": np.asarray(["c1", "c2"], dtype=object)},
        NA_action=NA_action,
    )[0]
    assert mat.shape == (2, 3)
    # According to this (and only this) function, NaN == NaN.
    np.testing.assert_array_equal(mat, [[1.0, 0.0, 10.0], [0.0, 1.0, np.nan]])

    # NA_action="raise"
    pytest.raises(
        PatsyError,
        build_design_matrices,
        [builder],
        {"x": [10.0, np.nan, 20.0], "c": np.asarray(["c1", "c2", None], dtype=object)},
        NA_action="raise",
    )


def test_NA_drop_preserves_levels():
    # Even if all instances of some level are dropped, we still include it in
    # the output matrix (as an all-zeros column)
    data = {"x": [1.0, np.nan, 3.0], "c": ["c1", "c2", "c3"]}

    def iter_maker():
        yield data

    design_info = design_matrix_builders([make_termlist("x", "c")], iter_maker, 0)[0]

    assert design_info.column_names == ["c[c1]", "c[c2]", "c[c3]", "x"]

    (mat,) = build_design_matrices([design_info], data)

    assert mat.shape == (2, 4)
    assert np.array_equal(mat, [[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 3.0]])


def test_return_type_pandas():
    if not have_pandas:
        return

    data = pandas.DataFrame(
        {"x": [1, 2, 3], "y": [4, 5, 6], "a": ["a1", "a2", "a1"]}, index=[10, 20, 30]
    )

    def iter_maker():
        yield data

    (int_builder,) = design_matrix_builders([make_termlist([])], iter_maker, 0)
    (y_builder, x_builder) = design_matrix_builders(
        [make_termlist("y"), make_termlist("x")], iter_maker, eval_env=0
    )
    (x_a_builder,) = design_matrix_builders(
        [make_termlist("x", "a")], iter_maker, eval_env=0
    )
    (x_y_builder,) = design_matrix_builders(
        [make_termlist("x", "y")], iter_maker, eval_env=0
    )
    # Index compatibility is always checked for pandas input, regardless of
    # whether we're producing pandas output
    pytest.raises(
        PatsyError,
        build_design_matrices,
        [x_a_builder],
        {"x": data["x"], "a": data["a"][::-1]},
    )
    pytest.raises(
        PatsyError,
        build_design_matrices,
        [y_builder, x_builder],
        {"x": data["x"], "y": data["y"][::-1]},
    )

    # And we also check consistency between data.index and value indexes
    # Creating a mismatch between these is a bit tricky. We want a data object
    # such that isinstance(data, DataFrame), but data["x"].index !=
    # data.index.
    class CheatingDataFrame(pandas.DataFrame):
        def __getitem__(self, key):
            if key == "x":
                return pandas.DataFrame.__getitem__(self, key)[::-1]
            else:
                return pandas.DataFrame.__getitem__(self, key)

    pytest.raises(
        PatsyError, build_design_matrices, [x_builder], CheatingDataFrame(data)
    )

    # A mix of pandas input and unindexed input is fine
    (mat,) = build_design_matrices([x_y_builder], {"x": data["x"], "y": [40, 50, 60]})
    assert np.allclose(mat, [[1, 40], [2, 50], [3, 60]])

    # with return_type="dataframe", we get out DataFrames with nice indices
    # and nice column names and design_info
    y_df, x_df = build_design_matrices(
        [y_builder, x_builder], data, return_type="dataframe"
    )
    assert isinstance(y_df, pandas.DataFrame)
    assert isinstance(x_df, pandas.DataFrame)
    assert np.array_equal(y_df, [[4], [5], [6]])
    assert np.array_equal(x_df, [[1], [2], [3]])
    assert np.array_equal(y_df.index, [10, 20, 30])
    assert np.array_equal(x_df.index, [10, 20, 30])
    assert np.array_equal(y_df.columns, ["y"])
    assert np.array_equal(x_df.columns, ["x"])
    assert y_df.design_info.column_names == ["y"]
    assert x_df.design_info.column_names == ["x"]
    assert y_df.design_info.term_names == ["y"]
    assert x_df.design_info.term_names == ["x"]
    # Same with mix of pandas and unindexed info, even if in different
    # matrices
    y_df, x_df = build_design_matrices(
        [y_builder, x_builder],
        {"y": [7, 8, 9], "x": data["x"]},
        return_type="dataframe",
    )
    assert isinstance(y_df, pandas.DataFrame)
    assert isinstance(x_df, pandas.DataFrame)
    assert np.array_equal(y_df, [[7], [8], [9]])
    assert np.array_equal(x_df, [[1], [2], [3]])
    assert np.array_equal(y_df.index, [10, 20, 30])
    assert np.array_equal(x_df.index, [10, 20, 30])
    assert np.array_equal(y_df.columns, ["y"])
    assert np.array_equal(x_df.columns, ["x"])
    assert y_df.design_info.column_names == ["y"]
    assert x_df.design_info.column_names == ["x"]
    assert y_df.design_info.term_names == ["y"]
    assert x_df.design_info.term_names == ["x"]
    # Check categorical works for carrying index too
    (x_a_df,) = build_design_matrices(
        [x_a_builder], {"x": [-1, -2, -3], "a": data["a"]}, return_type="dataframe"
    )
    assert isinstance(x_a_df, pandas.DataFrame)
    assert np.array_equal(x_a_df, [[1, 0, -1], [0, 1, -2], [1, 0, -3]])
    assert np.array_equal(x_a_df.index, [10, 20, 30])
    # And if we have no indexed input, then we let pandas make up an index as
    # per its usual rules:
    (x_y_df,) = build_design_matrices(
        [x_y_builder], {"y": [7, 8, 9], "x": [10, 11, 12]}, return_type="dataframe"
    )
    assert isinstance(x_y_df, pandas.DataFrame)
    assert np.array_equal(x_y_df, [[10, 7], [11, 8], [12, 9]])
    assert np.array_equal(x_y_df.index, [0, 1, 2])

    # If 'data' is a DataFrame, then that suffices, even if no factors are
    # available.
    (int_df,) = build_design_matrices([int_builder], data, return_type="dataframe")
    assert isinstance(int_df, pandas.DataFrame)
    assert np.array_equal(int_df, [[1], [1], [1]])
    assert int_df.index.equals(pandas.Index([10, 20, 30]))

    import patsy.build

    had_pandas = patsy.build.have_pandas
    try:
        patsy.build.have_pandas = False
        # return_type="dataframe" gives a nice error if pandas is not available
        pytest.raises(
            PatsyError,
            build_design_matrices,
            [x_builder],
            {"x": [1, 2, 3]},
            return_type="dataframe",
        )
    finally:
        patsy.build.have_pandas = had_pandas

    (x_df,) = build_design_matrices(
        [x_a_builder],
        {"x": [1.0, np.nan, 3.0], "a": np.asarray([None, "a2", "a1"], dtype=object)},
        NA_action="drop",
        return_type="dataframe",
    )
    assert x_df.index.equals(pandas.Index([2]))


def test_data_mismatch():
    test_cases_twoway = [
        # Data type mismatch
        ([1, 2, 3], [True, False, True]),
        (
            C(["a", "b", "c"], levels=["c", "b", "a"]),
            C(["a", "b", "c"], levels=["a", "b", "c"]),
        ),
        # column number mismatches
        ([[1], [2], [3]], [[1, 1], [2, 2], [3, 3]]),
        ([[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[1, 1], [2, 2], [3, 3]]),
    ]
    test_cases_oneway = [
        ([1, 2, 3], ["a", "b", "c"]),
        ([1, 2, 3], C(["a", "b", "c"])),
        ([True, False, True], C(["a", "b", "c"])),
        ([True, False, True], ["a", "b", "c"]),
    ]
    setup_predict_only = [
        # This is not an error if both are fed in during make_builders, but it
        # is an error to pass one to make_builders and the other to
        # make_matrices.
        (["a", "b", "c"], ["a", "b", "d"]),
    ]
    termlist = make_termlist(["x"])

    def t_incremental(data1, data2):
        def iter_maker():
            yield {"x": data1}
            yield {"x": data2}

        try:
            builders = design_matrix_builders([termlist], iter_maker, 0)
            build_design_matrices(builders, {"x": data1})
            build_design_matrices(builders, {"x": data2})
        except PatsyError:
            pass
        else:
            raise AssertionError

    def t_setup_predict(data1, data2):
        def iter_maker():
            yield {"x": data1}

        builders = design_matrix_builders([termlist], iter_maker, 0)
        pytest.raises(PatsyError, build_design_matrices, builders, {"x": data2})

    for a, b in test_cases_twoway:
        t_incremental(a, b)
        t_incremental(b, a)
        t_setup_predict(a, b)
        t_setup_predict(b, a)
    for a, b in test_cases_oneway:
        t_incremental(a, b)
        t_setup_predict(a, b)
    for a, b in setup_predict_only:
        t_setup_predict(a, b)
        t_setup_predict(b, a)

    pytest.raises(
        PatsyError, make_matrix, {"x": [1, 2, 3], "y": [1, 2, 3, 4]}, 2, [["x"], ["y"]]
    )


def test_data_independent_builder():
    data = {"x": [1, 2, 3]}

    def iter_maker():
        yield data

    # Trying to build a matrix that doesn't depend on the data at all is an
    # error, if:
    # - the index argument is not given
    # - the data is not a DataFrame
    # - there are no other matrices
    null_builder = design_matrix_builders([make_termlist()], iter_maker, 0)[0]
    pytest.raises(PatsyError, build_design_matrices, [null_builder], data)

    intercept_builder = design_matrix_builders(
        [make_termlist([])], iter_maker, eval_env=0
    )[0]
    pytest.raises(PatsyError, build_design_matrices, [intercept_builder], data)

    pytest.raises(
        PatsyError, build_design_matrices, [null_builder, intercept_builder], data
    )

    # If data is a DataFrame, it sets the number of rows.
    if have_pandas:
        int_m, null_m = build_design_matrices(
            [intercept_builder, null_builder], pandas.DataFrame(data)
        )
        assert np.allclose(int_m, [[1], [1], [1]])
        assert null_m.shape == (3, 0)

    # If there are other matrices that do depend on the data, we make the
    # data-independent matrices have the same number of rows.
    x_termlist = make_termlist(["x"])

    builders = design_matrix_builders(
        [x_termlist, make_termlist()], iter_maker, eval_env=0
    )
    x_m, null_m = build_design_matrices(builders, data)
    assert np.allclose(x_m, [[1], [2], [3]])
    assert null_m.shape == (3, 0)

    builders = design_matrix_builders(
        [x_termlist, make_termlist([])], iter_maker, eval_env=0
    )
    x_m, null_m = build_design_matrices(builders, data)
    x_m, intercept_m = build_design_matrices(builders, data)
    assert np.allclose(x_m, [[1], [2], [3]])
    assert np.allclose(intercept_m, [[1], [1], [1]])


def test_same_factor_in_two_matrices():
    data = {"x": [1, 2, 3], "a": ["a1", "a2", "a1"]}

    def iter_maker():
        yield data

    t1 = make_termlist(["x"])
    t2 = make_termlist(["x", "a"])
    builders = design_matrix_builders([t1, t2], iter_maker, eval_env=0)
    m1, m2 = build_design_matrices(builders, data)
    check_design_matrix(m1, 1, t1, column_names=["x"])
    assert np.allclose(m1, [[1], [2], [3]])
    check_design_matrix(m2, 2, t2, column_names=["x:a[a1]", "x:a[a2]"])
    assert np.allclose(m2, [[1, 0], [0, 2], [3, 0]])


def test_eval_env_type_builder():
    data = {"x": [1, 2, 3]}

    def iter_maker():
        yield data

    pytest.raises(
        TypeError, design_matrix_builders, [make_termlist("x")], iter_maker, "foo"
    )


def test_categorical():
    data_strings = {"a": ["a1", "a2", "a1"]}
    data_categ = {"a": C(["a2", "a1", "a2"])}
    datas = [data_strings, data_categ]
    if have_pandas_categorical:
        data_pandas = {"a": pandas.Categorical(["a1", "a2", "a2"])}
        datas.append(data_pandas)

    def t(data1, data2):
        def iter_maker():
            yield data1

        builders = design_matrix_builders(
            [make_termlist(["a"])], iter_maker, eval_env=0
        )
        build_design_matrices(builders, data2)

    for data1 in datas:
        for data2 in datas:
            t(data1, data2)


def test_contrast():
    from patsy.contrasts import ContrastMatrix, Sum

    values = ["a1", "a3", "a1", "a2"]

    # No intercept in model, full-rank coding of 'a'
    m = make_matrix(
        {"a": C(values)}, 3, [["a"]], column_names=["a[a1]", "a[a2]", "a[a3]"]
    )

    assert np.allclose(m, [[1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]])

    for s in (Sum, Sum()):
        m = make_matrix(
            {"a": C(values, s)},
            3,
            [["a"]],
            column_names=["a[mean]", "a[S.a1]", "a[S.a2]"],
        )
        # Output from R
        assert np.allclose(m, [[1, 1, 0], [1, -1, -1], [1, 1, 0], [1, 0, 1]])

    m = make_matrix(
        {"a": C(values, Sum(omit=0))},
        3,
        [["a"]],
        column_names=["a[mean]", "a[S.a2]", "a[S.a3]"],
    )
    # Output from R
    assert np.allclose(m, [[1, -1, -1], [1, 0, 1], [1, -1, -1], [1, 1, 0]])

    # Intercept in model, non-full-rank coding of 'a'
    m = make_matrix(
        {"a": C(values)},
        3,
        [[], ["a"]],
        column_names=["Intercept", "a[T.a2]", "a[T.a3]"],
    )

    assert np.allclose(m, [[1, 0, 0], [1, 0, 1], [1, 0, 0], [1, 1, 0]])

    for s in (Sum, Sum()):
        m = make_matrix(
            {"a": C(values, s)},
            3,
            [[], ["a"]],
            column_names=["Intercept", "a[S.a1]", "a[S.a2]"],
        )
        # Output from R
        assert np.allclose(m, [[1, 1, 0], [1, -1, -1], [1, 1, 0], [1, 0, 1]])

    m = make_matrix(
        {"a": C(values, Sum(omit=0))},
        3,
        [[], ["a"]],
        column_names=["Intercept", "a[S.a2]", "a[S.a3]"],
    )
    # Output from R
    assert np.allclose(m, [[1, -1, -1], [1, 0, 1], [1, -1, -1], [1, 1, 0]])

    # Weird ad hoc less-than-full-rank coding of 'a'
    m = make_matrix(
        {"a": C(values, [[7, 12], [2, 13], [8, -1]])},
        2,
        [["a"]],
        column_names=["a[custom0]", "a[custom1]"],
    )
    assert np.allclose(m, [[7, 12], [8, -1], [7, 12], [2, 13]])

    m = make_matrix(
        {
            "a": C(
                values, ContrastMatrix([[7, 12], [2, 13], [8, -1]], ["[foo]", "[bar]"])
            )
        },
        2,
        [["a"]],
        column_names=["a[foo]", "a[bar]"],
    )
    assert np.allclose(m, [[7, 12], [8, -1], [7, 12], [2, 13]])


def test_DesignInfo_subset():
    # For each combination of:
    #   formula, term names, term objects, mixed term name and term objects
    # check that results match subset of full build
    # and that removed variables don't hurt
    all_data = {"x": [1, 2], "y": [[3.1, 3.2], [4.1, 4.2]], "z": [5, 6]}
    all_terms = make_termlist("x", "y", "z")

    def iter_maker():
        yield all_data

    all_builder = design_matrix_builders([all_terms], iter_maker, 0)[0]
    full_matrix = build_design_matrices([all_builder], all_data)[0]

    def t(which_terms, variables, columns):
        sub_design_info = all_builder.subset(which_terms)
        sub_data = {}
        for variable in variables:
            sub_data[variable] = all_data[variable]
        sub_matrix = build_design_matrices([sub_design_info], sub_data)[0]
        sub_full_matrix = full_matrix[:, columns]
        if not isinstance(which_terms, str):
            assert len(which_terms) == len(sub_design_info.terms)
        assert np.array_equal(sub_matrix, sub_full_matrix)

    t("~ 0 + x + y + z", ["x", "y", "z"], slice(None))
    t(["x", "y", "z"], ["x", "y", "z"], slice(None))
    t(all_terms, ["x", "y", "z"], slice(None))
    t([all_terms[0], "y", all_terms[2]], ["x", "y", "z"], slice(None))

    t("~ 0 + x + z", ["x", "z"], [0, 3])
    t(["x", "z"], ["x", "z"], [0, 3])
    t([all_terms[0], all_terms[2]], ["x", "z"], [0, 3])
    t([all_terms[0], "z"], ["x", "z"], [0, 3])

    t("~ 0 + z + x", ["x", "z"], [3, 0])
    t(["z", "x"], ["x", "z"], [3, 0])
    t([all_terms[2], all_terms[0]], ["x", "z"], [3, 0])
    t([all_terms[2], "x"], ["x", "z"], [3, 0])

    t("~ 0 + y", ["y"], [1, 2])
    t(["y"], ["y"], [1, 2])
    t([all_terms[1]], ["y"], [1, 2])

    # Formula can't have a LHS
    pytest.raises(PatsyError, all_builder.subset, "a ~ a")
    # Term must exist
    pytest.raises(KeyError, all_builder.subset, "~ asdf")
    pytest.raises(KeyError, all_builder.subset, ["asdf"])
    pytest.raises(KeyError, all_builder.subset, [Term(["asdf"])])

    # Also check for a minimal DesignInfo (column names only)
    min_di = DesignInfo(["a", "b", "c"])
    min_di_subset = min_di.subset(["c", "a"])
    assert min_di_subset.column_names == ["c", "a"]
    assert min_di_subset.terms is None


def test_C_prob_point_mass_matches_C():
    # A point distribution over a single category must reproduce ordinary C
    # exactly, for full and reduced codings and for several contrasts.
    labels = ["a1", "a3", "a1", "a2"]
    probs = [{label: 1.0} for label in labels]

    def dmatrix_safe(formula, data):
        from patsy.highlevel import dmatrix

        return dmatrix(formula, data)

    for contrast_suffix in ["", ", Sum", ", Helmert", ", Poly", ", Diff"]:
        for prefix in ["", "0 + "]:
            prob_m = dmatrix_safe(
                prefix + "C_prob(p%s)" % contrast_suffix, {"p": probs}
            )
            hard_m = dmatrix_safe(prefix + "C(a%s)" % contrast_suffix, {"a": labels})
            assert np.allclose(prob_m, hard_m)
            # Column names only differ by the factor name.
            assert [
                name.replace("C_prob(p", "C(a")
                for name in prob_m.design_info.column_names
            ] == hard_m.design_info.column_names


def test_C_prob_normalization():
    # Weights are normalized per row; scaling a row's weights has no effect.
    from patsy.highlevel import dmatrix

    p = [{"a": 1.0, "b": 3.0}, {"a": 200.0, "b": 600.0}]
    m = dmatrix("0 + C_prob(p, levels=['a', 'b'])", {"p": p})
    assert np.allclose(m, [[0.25, 0.75], [0.25, 0.75]])

    # Fractional weights work too.
    m = dmatrix("0 + C_prob(p, levels=['a', 'b'])", {"p": [{"a": 0.2, "b": 0.3}]})
    assert np.allclose(m, [[0.4, 0.6]])


def test_C_prob_interaction():
    # Interactions of probability factors multiply their encoded columns, just
    # like ordinary categorical factors, and match the equivalent hard-label
    # design with the same point masses.
    from patsy.highlevel import dmatrix

    data = {
        "p": [{"a": 1.0}, {"b": 1.0}, {"a": 1.0}],
        "q": [{"x": 1.0}, {"y": 1.0}, {"y": 1.0}],
        "c": ["a", "b", "a"],
        "d": ["x", "y", "y"],
        "z": [1.0, 2.0, 3.0],
    }
    prob = dmatrix("C_prob(p) + C_prob(q) + C_prob(p):C_prob(q) + C_prob(p):z", data)
    hard = dmatrix("C(c) + C(d) + C(c):C(d) + C(c):z", data)
    assert prob.design_info.column_names == [
        "Intercept",
        "C_prob(p)[T.b]",
        "C_prob(q)[T.y]",
        "C_prob(p)[T.b]:C_prob(q)[T.y]",
        "C_prob(p)[a]:z",
        "C_prob(p)[b]:z",
    ]
    assert np.allclose(prob, hard)

    # A genuinely soft interaction multiplies the two probability encodings.
    soft = dmatrix(
        "0 + C_prob(p, levels=['a', 'b']):C_prob(q, levels=['x', 'y'])",
        {
            "p": [{"a": 0.5, "b": 0.5}],
            "q": [{"x": 0.25, "y": 0.75}],
        },
    )
    assert soft.design_info.column_names == [
        "C_prob(p, levels=['a', 'b'])[a]:C_prob(q, levels=['x', 'y'])[x]",
        "C_prob(p, levels=['a', 'b'])[b]:C_prob(q, levels=['x', 'y'])[x]",
        "C_prob(p, levels=['a', 'b'])[a]:C_prob(q, levels=['x', 'y'])[y]",
        "C_prob(p, levels=['a', 'b'])[b]:C_prob(q, levels=['x', 'y'])[y]",
    ]
    assert np.allclose(soft, [[0.125, 0.125, 0.375, 0.375]])


def test_C_prob_level_discovery_and_explicit_levels():
    from patsy.highlevel import dmatrix

    # Levels come from categories with positive mass, and are sorted.
    m = dmatrix("0 + C_prob(p)", {"p": [{"b": 1.0}, {"a": 0.5, "b": 0.5}]})
    assert m.design_info.column_names == ["C_prob(p)[a]", "C_prob(p)[b]"]

    # Categories with zero mass do not create levels.
    m = dmatrix("0 + C_prob(p)", {"p": [{"a": 1.0, "c": 0.0}]})
    assert m.design_info.column_names == ["C_prob(p)[a]"]

    # Explicit levels override discovery and ordering.
    m = dmatrix("0 + C_prob(p, levels=['b', 'a'])", {"p": [{"a": 1.0}, {"b": 1.0}]})
    assert m.design_info.column_names == [
        "C_prob(p, levels=['b', 'a'])[b]",
        "C_prob(p, levels=['b', 'a'])[a]",
    ]
    assert np.allclose(m, [[0, 1], [1, 0]])

    # Explicit contrast is honoured.
    m = dmatrix("C_prob(p, Sum)", {"p": [{"a": 1.0}, {"b": 1.0}]})
    assert m.design_info.column_names == ["Intercept", "C_prob(p, Sum)[S.a]"]


def test_C_prob_prediction_unknown_category():
    from patsy.highlevel import dmatrix

    train = {"p": [{"a": 1.0}, {"b": 1.0}, {"a": 0.5, "b": 0.5}]}
    design_info = dmatrix("C_prob(p)", train).design_info

    # Unknown category with positive mass is an error, not dropped.
    pytest.raises(
        PatsyError,
        build_design_matrices,
        [design_info],
        {"p": [{"a": 1.0}, {"c": 1.0}]},
    )
    # ... but one with zero mass is harmless.
    (m,) = build_design_matrices([design_info], {"p": [{"a": 1.0}, {"c": 0.0}]})
    assert np.allclose(m, [[1, 0], [1, 0]])


def test_C_prob_NA_handling():
    from patsy.highlevel import dmatrix

    design_info = dmatrix(
        "C_prob(p, levels=['a', 'b'])", {"p": [{"a": 1.0}, {"b": 1.0}]}
    ).design_info

    # Empty mapping and zero total are categorical missing.
    data = {"p": [{"a": 1.0}, {}, {"a": 0, "b": 0}, {"b": 1.0}]}
    (m,) = build_design_matrices([design_info], data)
    assert m.shape == (2, 2)
    assert np.allclose(m, [[1, 0], [1, 1]])

    pytest.raises(
        PatsyError, build_design_matrices, [design_info], data, NA_action="raise"
    )


def test_C_prob_invalid_weights():
    from patsy.highlevel import dmatrix

    for bad in ({"a": -1.0}, {"a": np.nan}, {"a": np.inf}, {"a": "x"}):
        pytest.raises(PatsyError, dmatrix, "C_prob(p)", {"p": [bad]})


def test_C_prob_DesignInfo_subset():
    from patsy.highlevel import dmatrix

    train = {
        "p": [{"a": 1.0}, {"b": 1.0}, {"a": 0.5, "b": 0.5}],
        "x": [1.0, 2.0, 3.0],
    }
    full = dmatrix("1 + C_prob(p) + x", train)
    sub = full.design_info.subset(["Intercept", "C_prob(p)"])
    assert sub.column_names == full.design_info.column_names[:2]
    (m,) = build_design_matrices([sub], train)
    assert np.allclose(m, full[:, :2])

    # The removed factor ('x') need not be present when predicting from the
    # subset.
    (m2,) = build_design_matrices([sub], {"p": train["p"]})
    assert np.allclose(m2, full[:, :2])

    # ... and term objects work as well as names.
    term_objs = [t for t in full.design_info.terms if t.name() == "C_prob(p)"]
    sub2 = full.design_info.subset(term_objs)
    assert sub2.column_names == full.design_info.column_names[1:2]
    (m3,) = build_design_matrices([sub2], {"p": train["p"]})
    assert np.allclose(m3, full[:, 1:2])


def test_C_prob_pandas_index():
    if not have_pandas:
        return
    from patsy.highlevel import dmatrix

    df = pandas.DataFrame(
        {"p": [{"a": 1.0}, {}, {"b": 1.0}], "x": [1.0, 2.0, 3.0]},
        index=[7, 8, 9],
    )
    out = dmatrix("C_prob(p) + x", df, return_type="dataframe")
    assert out.index.tolist() == [7, 9]
    assert np.allclose(out.values, [[1, 0, 1], [1, 1, 3]])


def test_C_prob_misc():
    from patsy.highlevel import dmatrix

    # Non-string categories work, and are sorted with the usual rules.
    m = dmatrix("0 + C_prob(p)", {"p": [{1: 1.0}, {2: 1.0}]})
    assert m.design_info.column_names == ["C_prob(p)[1]", "C_prob(p)[2]"]
    m = dmatrix("0 + C_prob(p)", {"p": [{True: 1.0}, {False: 1.0}]})
    assert m.design_info.column_names == ["C_prob(p)[False]", "C_prob(p)[True]"]

    # Accidentally passing raw mappings as a plain factor gives a helpful
    # error rather than a confusing one.
    pytest.raises(PatsyError, dmatrix, "p", {"p": [{"a": 1.0}]})

    # Empty data with explicit levels still builds an empty (zero-row)
    # matrix, matching C's NA-drop behaviour.
    m = dmatrix("C_prob(p, levels=['a', 'b'])", {"p": [{}, {}]})
    assert m.shape == (0, 2)
    assert m.design_info.column_names == [
        "Intercept",
        "C_prob(p, levels=['a', 'b'])[T.b]",
    ]


def test_C_prob_misuse():
    from patsy.categorical import C_prob
    from patsy.highlevel import dmatrix

    # Wrapping probability data in C() is a mistake and is reported clearly.
    pytest.raises(
        PatsyError,
        dmatrix,
        "0 + C(p)",
        {"p": C_prob([{"a": 1.0}])},
    )
