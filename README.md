

# generate-tests-cpp


This repository offers various illustrations on prompting open-source Large Language Models (LLMs) 
to generate C++ unit tests:

- `example_function.py` holds the code snippet used for demonstration.
- `templates.py` comprises the prompt templates tailored for the open-source models.
- `run_model_name.py` includes the scripts for executing the generation process using the example provided. Currently, only WizardCoder and DeepSeek Coder base models are supported.

## Running the test generation

To run the generation, just run the following
```shell
python run_model_name.py
```

For instance, in the case of WizardCoder, you have to run:
```shell
python run_wizardcoder.py
```

The output should have the following shape (concatenation of the prompt and the tests):
```text
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Write 5 google tests for this function:
bool has_close_elements(vector<float> numbers, float threshold){
    int i,j;
    for (i=0;i<numbers.size();i++)
        for (j=i+1;j<numbers.size();j++)
            if (abs(numbers[i]-numbers[j])<threshold)
                return true;
    return false;
}

### Response:
TEST(has_close_elements, basic_test) {
    vector<float> numbers = {1.0, 2.0, 3.0, 4.0, 5.0};
    float threshold = 1.0;
    EXPECT_TRUE(has_close_elements(numbers, threshold));
}

TEST(has_close_elements, edge_test) {
    vector<float> numbers = {1.0, 2.0, 3.0, 4.0, 5.0};
    float threshold = 0.5;
    EXPECT_FALSE(has_close_elements(numbers, threshold));
}

TEST(has_close_elements, middle_test) {
    vector<float> numbers = {1.0, 2.0, 3.0, 4.0, 5.0};
    float threshold = 2.0;
    EXPECT_TRUE(has_close_elements(numbers, threshold));
}

TEST(has_close_elements, all_close_test) {
    vector<float> numbers = {1.0, 2.0, 3.0, 4.0, 5.0};
    float threshold = 0.1;
    EXPECT_TRUE(has_close_elements(numbers, threshold));
}

TEST(has_close_elements, all_different_test) {
    vector<float> numbers = {1.0, 2.0, 3.0, 4.0, 5.1};
    float threshold = 0.1;
    EXPECT_FALSE(has_close_elements(numbers, threshold));
}
```

## Running the HumanEval generation

The `run_wizardcoder_humaneval.py` script generates 5 functions for the first prompt of the HumanEval dataset.
It also executes the code snippets using the test sets of the same dataset.

```shell
python run_wizardcoder_humaneval.py
```