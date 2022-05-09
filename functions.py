"""
********************************************************************************
functions to be learned
********************************************************************************
"""

import numpy as np
import tensorflow as tf

def func0(t):
    y = np.sin(t)
    return y
def func0_tf(t):
    y = tf.sin(t)
    return y

def func1(t):
    y = .5 * np.sin(2. * np.pi * 3. * t) \
        + .5 * np.sin(2. * np.pi * 10. * t) \
        + .5 * np.sin(2. * np.pi * 45. * t)
    return y
def func1_tf(t):
    y = .5 * tf.sin(2. * np.pi * 3. * t) \
        + .5 * tf.sin(2. * np.pi * 10. * t) \
        + .5 * tf.sin(2. * np.pi * 45. * t)
    return y

def func2(t):
    y = np.sin(2. * np.pi * t) \
        + 1. / 3. * np.sin(2. * np.pi * 3. * t) \
        + 1. / 5. * np.sin(2. * np.pi * 5. * t) \
        + 1. / 7. * np.sin(2. * np.pi * 7. * t) \
        + 1. / 9. * np.sin(2. * np.pi * 9. * t) \
        + 1. / 11. * np.sin(2. * np.pi * 11. * t) \
        + 1. / 13. * np.sin(2. * np.pi * 13. * t) \
        + 1. / 15. * np.sin(2. * np.pi * 15. * t)
    y *= 4 / np.pi
    return y
def func2_tf(t):
    y = tf.sin(2. * np.pi * t) \
        + 1. / 3. * tf.sin(2. * np.pi * 3. * t) \
        + 1. / 5. * tf.sin(2. * np.pi * 5. * t) \
        + 1. / 7. * tf.sin(2. * np.pi * 7. * t) \
        + 1. / 9. * tf.sin(2. * np.pi * 9. * t) \
        + 1. / 11. * tf.sin(2. * np.pi * 11. * t) \
        + 1. / 13. * tf.sin(2. * np.pi * 13. * t) \
        + 1. / 15. * tf.sin(2. * np.pi * 15. * t)
    y *= 4 / np.pi
    return y

