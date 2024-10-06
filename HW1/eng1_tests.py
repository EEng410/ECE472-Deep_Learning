# # /bin/env python3.8

import pytest


def test_additivity():
    import tensorflow as tf

    from eng1 import Linear

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_outputs = 1
    M = 4

    linear = Linear(num_inputs, num_outputs, M)

    a = rng.normal(shape=[1, M])
    b = rng.normal(shape=[1, M])

    tf.debugging.assert_near(linear(a + b), linear(a) + linear(b), summarize=2)


def test_homogeneity():
    import tensorflow as tf

    from eng1 import Linear

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_outputs = 1
    M = 4
    num_test_cases = 100

    linear = Linear(num_inputs, num_outputs, M)

    a = rng.normal(shape=[1, M])
    b = rng.normal(shape=[num_test_cases, 1])

    tf.debugging.assert_near(linear(a * b), linear(a) * b, summarize=2)


@pytest.mark.parametrize("num_outputs", [1, 16, 128])
def test_dimensionality(num_outputs):
    import tensorflow as tf

    from eng1 import Linear

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    M = 4

    linear = Linear(num_inputs, num_outputs, M)

    a = rng.normal(shape=[1, M])
    z = linear(a)

    tf.assert_equal(tf.shape(z)[-1], num_outputs)


@pytest.mark.parametrize("bias", [True, False])
def test_trainable(bias):
    import tensorflow as tf

    from eng1 import Linear

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_outputs = 1
    M = 4

    linear = Linear(num_inputs, num_outputs, M, bias=bias)

    a = rng.normal(shape=[1, M])

    with tf.GradientTape() as tape:
        z = linear(a)
        loss = tf.math.reduce_mean(z**2)

    grads = tape.gradient(loss, linear.trainable_variables)

    for grad, var in zip(grads, linear.trainable_variables):
        tf.debugging.check_numerics(grad, message=f"{var.name}: ")
        tf.debugging.assert_greater(tf.math.abs(grad), 0.0)

    assert len(grads) == len(linear.trainable_variables)

    if bias:
        assert len(grads) == 2
    else:
        assert len(grads) == 1


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [([1000, 1000], [100, 100]), ([1000, 100], [100, 100]), ([100, 1000], [100, 100])],
)
def test_init_properties(a_shape, b_shape):
    import tensorflow as tf

    from eng1 import Linear

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs_a, num_outputs_a = a_shape
    num_inputs_b, num_outputs_b = b_shape
    M = 4

    linear_a = Linear(num_inputs_a, num_outputs_a, M, bias=False)
    linear_b = Linear(num_inputs_b, num_outputs_b, M,  bias=False)

    std_a = tf.math.reduce_std(linear_a.w)
    std_b = tf.math.reduce_std(linear_b.w)

    tf.debugging.assert_less(std_a, std_b)


def test_bias():
    import tensorflow as tf

    from eng1 import Linear

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)
    M = 4
    linear_with_bias = Linear(1, 1, M, bias=True)
    assert hasattr(linear_with_bias, "b")

    linear_with_bias = Linear(1, 1, M, bias=False)
    assert not hasattr(linear_with_bias, "b")

@pytest.mark.parametrize(
    "batch_size, M",
    [(1, 1), (2, 2), (20, 4), (50, 10)],
)
def test_Basis_Expansion_shapes(batch_size, M):
    import tensorflow as tf
    
    from eng1 import BasisExpansion

    num_inputs = 1
    # M = 4
    # batch_size = 20
    basis = BasisExpansion(num_inputs, M)

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    a = rng.normal(shape=[batch_size, 1])
    output_shape = batch_size, M
    phi = basis(a)
    tf.assert_equal(tf.shape(phi), output_shape)

@pytest.mark.parametrize(
    "batch_size, M",
    [(1, 1), (2, 2), (20, 4), (50, 10)],
)
def test_Basis_Expansion_Linear_integration(batch_size, M):
    import tensorflow as tf
    
    from eng1 import BasisExpansion
    from eng1 import Linear
    num_inputs = 1
    num_outputs = 1
    # M = 4
    # batch_size = 1  
    basis = BasisExpansion(num_inputs, M)
    linear = Linear(num_inputs, num_outputs, M)
    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    a = rng.normal(shape=[batch_size, 1])
    output_shape = batch_size, num_outputs
    phi = basis(a)
    y = linear(phi)

    tf.assert_equal(tf.shape(y), output_shape)

# Can't really test for the means and variances actually matching because we did not perform the computation in the strict functional form of the normal distribution. 

# def test_Basis_Expansion_init_parameters():
#     import tensorflow as tf
    
#     from eng1 import BasisExpansion
#     from eng1 import Linear
#     num_inputs = 1
#     num_outputs = 1
#     M = 4
#     batch_size = 1000  
#     basis = BasisExpansion(num_inputs, M)
#     # linear = Linear(num_inputs, num_outputs, M)
#     rng = tf.random.get_global_generator()
#     rng.reset_from_seed(2384230948)

#     means = basis.mu
#     std = basis.sigma

#     a = rng.normal(shape=[batch_size, 1])
#     # output_shape = batch_size, num_outputs
#     phi = basis(a)
#     breakpoint()

