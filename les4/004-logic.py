#!/usr/bin/env python
# coding: utf-8

# ## Логическая операция AND

# In[ ]:


import pandas as pd


# In[ ]:


# TODO: подберите weight1, weight2, and bias
# так, чтобы выполнялась операция AND 
# для входных данных [(0, 0), (0, 1), (1, 0), (1, 1)]
weight1 = 0
weight2 = 0
bias = 0

# уравнение гиперплоскости:
# y(x) = - x * weight1 / weight2 - bias / weight2

# входные данные и разметка
test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
correct_outputs = [False, False, False, True]
outputs = []

# вычисляем предсказания
for test_input, correct_output in zip(test_inputs, correct_outputs):
    linear_combination = weight1 * test_input[0] + weight2 * test_input[1] + bias
    # результат линейной комбнации сравниваем с нулем
    output = int(linear_combination >= 0)
    is_correct_string = 'Yes' if output == correct_output else 'No'
    outputs.append([test_input[0],
                    test_input[1],
                    linear_combination, 
                    output,
                    is_correct_string])

# выводим результат
output_frame = pd.DataFrame(outputs, columns=['Input 1', 
                                              '  Input 2', 
                                              '  Linear Combination', 
                                              '  Activation Output', 
                                              '  Is Correct'])
print(output_frame.to_string(index = False))


# ## Логическая операция NOT

# In[ ]:


# TODO: подберите weight1, weight2, and bias
# так, чтобы выполнялась операция NOT 
# для входных данных [(0, 0), (0, 1), (1, 0), (1, 1)]
weight1 = 0
weight2 = 0
bias = 0

# уравнение гиперплоскости:
# y(x) = - x * weight1 / weight2 - bias / weight2

# входные данные и разметка
test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
correct_outputs = [True, False, True, False]
outputs = []

# вычисляем предсказания
for test_input, correct_output in zip(test_inputs, correct_outputs):
    linear_combination = weight1 * test_input[0] + weight2 * test_input[1] + bias
    # результат линейной комбнации сравниваем с нулем
    output = int(linear_combination >= 0)
    is_correct_string = 'Yes' if output == correct_output else 'No'
    outputs.append([test_input[0],
                    test_input[1],
                    linear_combination, 
                    output, 
                    is_correct_string])

# выводим результат
output_frame = pd.DataFrame(outputs, columns=['Input 1',
                                              '  Input 2',
                                              '  Linear Combination',
                                              '  Activation Output', '  Is Correct'])
print(output_frame.to_string(index=False))

