# Project Description 

> adjusted from the main reference

## Introduction

A couple of years ago, J.K. Rowling (of Harry Potter fame) tried something interesting. She wrote a book, `The Cuckoo’s Calling,` under the name Robert Galbraith. The book received some good reviews, but no one paid much attention to it--until an anonymous tipster on Twitter said it was J.K. Rowling. The London Sunday Times enlisted two experts to compare the linguistic patterns of “Cuckoo” to Rowling’s “The Casual Vacancy,” as well as to books by several other authors. After the results of their analysis pointed strongly toward Rowling as the author, the Times directly asked the publisher if they were the same person, and the publisher confirmed. The book exploded in popularity overnight.

## Problem Description

Similar to the above, we have a set of emails, half of which were written by one person and the other half by another person at the same company . The **objective** is to classify the emails as written by one person or the other based only on the text of the email. Naive Bayes is to be used to classify emails by author.

## Data and Initial Code

- A list of strings. Each string is the text of an email, which has undergone some basic preprocessing
- An initial code to split the dataset into training and testing sets. 

## Dependencies and Run
- scikit-learn
- nltk : natural language toolkit

> initially `run startup.py` from ../tools/ to download related datasets

# Notes

One particular feature of Naive Bayes is that it’s a good algorithm for working with text classification. When dealing with text, it’s very common to treat each unique word as a feature, and since the typical person’s vocabulary is many thousands of words, this makes for a large number of features. The `relative simplicity` of the algorithm and the `independent features assumption` of Naive Bayes make it a strong performer for classifying texts. 
