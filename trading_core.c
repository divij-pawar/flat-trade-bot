// trading_core.c - C extension for critical trading bot functions

#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

// Calculate Simple Moving Average
static PyObject* calculate_sma(PyObject* self, PyObject* args) {
    PyArrayObject *price_array;
    int window;
    
    // Parse arguments: price array and window size
    if (!PyArg_ParseTuple(args, "Oi", &price_array, &window)) {
        return NULL;
    }
    
    // Get dimensions
    npy_intp size = PyArray_DIM(price_array, 0);
    
    // Create output array
    npy_intp dims[1] = {size};
    PyArrayObject *result = (PyArrayObject*)PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    
    // Calculate SMA
    double *prices = (double*)PyArray_DATA(price_array);
    double *sma = (double*)PyArray_DATA(result);
    
    double sum = 0.0;
    int i;
    
    for (i = 0; i < size; i++) {
        sum += prices[i];
        
        if (i >= window) {
            sum -= prices[i - window];
            sma[i] = sum / window;
        } else if (i == window - 1) {
            sma[i] = sum / window;
        } else {
            sma[i] = NAN;  // Not enough data for window
        }
    }
    
    return PyArray_Return(result);
}

// Calculate Exponential Moving Average
static PyObject* calculate_ema(PyObject* self, PyObject* args) {
    PyArrayObject *price_array;
    int span;
    
    // Parse arguments
    if (!PyArg_ParseTuple(args, "Oi", &price_array, &span)) {
        return NULL;
    }
    
    // Get dimensions
    npy_intp size = PyArray_DIM(price_array, 0);
    
    // Create output array
    npy_intp dims[1] = {size};
    PyArrayObject *result = (PyArrayObject*)PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    
    // Calculate EMA
    double *prices = (double*)PyArray_DATA(price_array);
    double *ema = (double*)PyArray_DATA(result);
    
    double alpha = 2.0 / (span + 1.0);
    int i;
    
    // First value is just the price
    ema[0] = prices[0];
    
    // Calculate EMA for remaining values
    for (i = 1; i < size; i++) {
        ema[i] = prices[i] * alpha + ema[i-1] * (1.0 - alpha);
    }
    
    return PyArray_Return(result);
}

// Calculate RSI (Relative Strength Index)
static PyObject* calculate_rsi(PyObject* self, PyObject* args) {
    PyArrayObject *price_array;
    int period;
    
    // Parse arguments
    if (!PyArg_ParseTuple(args, "Oi", &price_array, &period)) {
        return NULL;
    }
    
    // Get dimensions
    npy_intp size = PyArray_DIM(price_array, 0);
    if (size <= period) {
        PyErr_SetString(PyExc_ValueError, "Input array size must be greater than period");
        return NULL;
    }
    
    // Create output array
    npy_intp dims[1] = {size};
    PyArrayObject *result = (PyArrayObject*)PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    
    // Calculate RSI
    double *prices = (double*)PyArray_DATA(price_array);
    double *rsi = (double*)PyArray_DATA(result);
    
    double gain_sum = 0.0;
    double loss_sum = 0.0;
    double price_diff;
    int i;
    
    // Calculate initial average gain and loss
    for (i = 1; i <= period; i++) {
        price_diff = prices[i] - prices[i-1];
        if (price_diff > 0) {
            gain_sum += price_diff;
        } else {
            loss_sum -= price_diff; // Make loss positive
        }
        rsi[i-1] = NAN; // Not enough data yet
    }
    
    double avg_gain = gain_sum / period;
    double avg_loss = loss_sum / period;
    
    // Calculate first RSI
    if (avg_loss == 0) {
        rsi[period-1] = 100.0;
    } else {
        double rs = avg_gain / avg_loss;
        rsi[period-1] = 100.0 - (100.0 / (1.0 + rs));
    }
    
    // Calculate remaining RSIs using smoothed averages
    for (i = period; i < size; i++) {
        price_diff = prices[i] - prices[i-1];
        
        if (price_diff > 0) {
            avg_gain = (avg_gain * (period - 1) + price_diff) / period;
            avg_loss = (avg_loss * (period - 1)) / period;
        } else {
            avg_gain = (avg_gain * (period - 1)) / period;
            avg_loss = (avg_loss * (period - 1) - price_diff) / period;
        }
        
        if (avg_loss == 0) {
            rsi[i] = 100.0;
        } else {
            double rs = avg_gain / avg_loss;
            rsi[i] = 100.0 - (100.0 / (1.0 + rs));
        }
    }
    
    return PyArray_Return(result);
}

// Generate trading signals based on crossover strategy
static PyObject* generate_crossover_signals(PyObject* self, PyObject* args) {
    PyArrayObject *fast_ma, *slow_ma;
    
    // Parse arguments
    if (!PyArg_ParseTuple(args, "OO", &fast_ma, &slow_ma)) {
        return NULL;
    }
    
    // Get dimensions
    npy_intp size = PyArray_DIM(fast_ma, 0);
    if (size != PyArray_DIM(slow_ma, 0)) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must have the same size");
        return NULL;
    }
    
    // Create output array
    npy_intp dims[1] = {size};
    PyArrayObject *result = (PyArrayObject*)PyArray_ZEROS(1, dims, NPY_INT, 0);
    
    // Generate signals
    double *fast = (double*)PyArray_DATA(fast_ma);
    double *slow = (double*)PyArray_DATA(slow_ma);
    int *signals = (int*)PyArray_DATA(result);
    
    int i;
    
    signals[0] = 0; // No signal for first data point
    
    for (i = 1; i < size; i++) {
        // Check for crossover
        if (fast[i] > slow[i] && fast[i-1] <= slow[i-1]) {
            signals[i] = 1; // Buy signal
        } 
        else if (fast[i] < slow[i] && fast[i-1] >= slow[i-1]) {
            signals[i] = -1; // Sell signal
        }
        else {
            signals[i] = 0; // No signal
        }
    }
    
    return PyArray_Return(result);
}

// Risk management check
static PyObject* check_risk(PyObject* self, PyObject* args) {
    double capital, position_value, volatility, max_position_percent, max_volatility;
    
    // Parse arguments
    if (!PyArg_ParseTuple(args, "ddddd", &capital, &position_value, &volatility,
                          &max_position_percent, &max_volatility)) {
        return NULL;
    }
    
    // Check if we have enough capital
    if (capital < 1000.0) {
        return Py_BuildValue("is", 0, "Insufficient capital");
    }
    
    // Check if volatility is acceptable
    if (volatility > max_volatility) {
        return Py_BuildValue("is", 0, "Volatility too high");
    }
    
    // Position sizing check
    double max_position_size = capital * max_position_percent;
    if (position_value > max_position_size) {
        return Py_BuildValue("is", 0, "Position size too large");
    }
    
    // All checks passed
    return Py_BuildValue("is", 1, "Risk checks passed");
}

// Module's function table
static PyMethodDef TradingCoreMethods[] = {
    {"calculate_sma", calculate_sma, METH_VARARGS, "Calculate Simple Moving Average"},
    {"calculate_ema", calculate_ema, METH_VARARGS, "Calculate Exponential Moving Average"},
    {"calculate_rsi", calculate_rsi, METH_VARARGS, "Calculate Relative Strength Index"},
    {"generate_crossover_signals", generate_crossover_signals, METH_VARARGS, "Generate trading signals based on MA crossover"},
    {"check_risk", check_risk, METH_VARARGS, "Perform risk management checks"},
    {NULL, NULL, 0, NULL} // Sentinel
};

// Module definition
static struct PyModuleDef tradingcoremodule = {
    PyModuleDef_HEAD_INIT,
    "trading_core",
    "C extension module for trading bot core functions",
    -1,
    TradingCoreMethods
};

// Module initialization function
PyMODINIT_FUNC PyInit_trading_core(void) {
    PyObject *m;
    
    m = PyModule_Create(&tradingcoremodule);
    if (m == NULL)
        return NULL;
    
    // Import NumPy
    import_array();
    
    return m;
}