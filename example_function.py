FUNC = """bool has_close_elements(vector<float> numbers, float threshold){
    int i,j;
    for (i=0;i<numbers.size();i++)
        for (j=i+1;j<numbers.size();j++)
            if (abs(numbers[i]-numbers[j])<threshold)
                return true;
    return false;
}"""
FUNC_NAME = "has_close_elements"
DOCUMENTATION = "Check if in given vector of numbers, are any two numbers closer to each other than given threshold."
