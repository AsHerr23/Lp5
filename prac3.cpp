#include <omp.h>
#include <iostream>
#include <chrono>
using namespace std::chrono;
using namespace std;

// Display Array
void displayArray(int nums[], int length)
{
    cout << "Nums: [";
    for (int i = 0; i < length; i++) {
        cout << nums[i];
        if (i != length - 1)
            cout << ", ";
    }
    cout << "]" << endl;
}

// Parallel Min Operation
void minOperation(int nums[], int length)
{
    int minValue = nums[0];
#pragma omp parallel for reduction(min : minValue)
    for (int i = 0; i < length; i++) {
        minValue = (nums[i] < minValue) ? nums[i] : minValue;
    }
    cout << "Min value: " << minValue << endl;
}

// Parallel Max Operation
void maxOperation(int nums[], int length)
{
    int maxValue = nums[0];
#pragma omp parallel for reduction(max : maxValue)
    for (int i = 0; i < length; i++) {
        maxValue = (nums[i] > maxValue) ? nums[i] : maxValue;
    }
    cout << "Max value: " << maxValue << endl;
}

// Parallel Sum Operation
void sumOperation(int nums[], int length)
{
    int sum = 0;
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < length; i++) {
        sum += nums[i];
    }
    cout << "Sum: " << sum << endl;
}

// Parallel Average Operation
void avgOperation(int nums[], int length)
{
    float sum = 0;
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < length; i++) {
        sum += nums[i];
    }
    cout << "Average: " << (sum / length) << endl;
}

// Main Function
int main()
{
    int nums[] = {4, 6, 3, 2, 6, 7, 9, 2, 1, 6, 5};
    int length = sizeof(nums) / sizeof(int);

    auto start = high_resolution_clock::now();

    displayArray(nums, length);
    minOperation(nums, length);
    maxOperation(nums, length);
    sumOperation(nums, length);
    avgOperation(nums, length);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "\nExecution time: " << duration.count() << " microseconds" << endl;

    return 0;
}
