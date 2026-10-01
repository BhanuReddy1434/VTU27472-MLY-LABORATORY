# Task 1: Find-S and Candidate Elimination
# Based on the Machine Learning Techniques Laboratory Manual

positive_examples = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Cool', 'Change'],
    ['Sunny', 'Hot', 'High', 'Strong', 'Cool', 'Change']
]

negative_examples = [
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change'],
    ['Sunny', 'Warm', 'Normal', 'Weak', 'Cool', 'Same']
]

def find_s(examples):
    hypothesis = ['Ø'] * len(examples[0])
    for example in examples:
        for i, value in enumerate(example):
            if hypothesis[i] == 'Ø':
                hypothesis[i] = value
            elif hypothesis[i] != value:
                hypothesis[i] = '?'
    return hypothesis

def candidate_elimination(positive, negative):
    specific = ['Ø'] * len(positive[0])
    general = [['?'] * len(positive[0])]

    for example in positive:
        for i, value in enumerate(example):
            if specific[i] == 'Ø':
                specific[i] = value
            elif specific[i] != value:
                specific[i] = '?'

    for example in negative:
        for i, value in enumerate(example):
            if specific[i] != '?' and value != specific[i]:
                general[0][i] = specific[i]

    return specific, general

print("FIND-S Hypothesis:", find_s(positive_examples))
specific, general = candidate_elimination(positive_examples, negative_examples)
print("Candidate-Elimination Specific Hypothesis:", specific)
print("Candidate-Elimination General Hypothesis:", general)
