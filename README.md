## Loan Payoff Predictor

### Table of contents
1. [Introduction](#Introduction)
2. [Getting Started](#GettingStarted)
    1. [Dependency](#Dependencies)
    2. [Installation](#Installation)
    3. [Executing Program](#Execution)
3. [Authors](#Authors)
4. [License](#License)
5. [Acknowledgement](#Acknowledgement)
6. [Screenshots](#Screenshots)

## Introduction <a name='Introduction'></a>
Publicly available data from LendingClub.com. Lending Club connects people who need money (borrowers) with people who have money (investors). Hopefully, as an investor you would want to invest in people who showed a profile of having a high probability of paying you back.

We will use lending data from 2007-2010 and be trying to classify and predict whether or not the borrower paid back their loan in full. You can download the data from here.

Here are what the columns represent:

1. credit.policy: 1 if the customer meets the credit underwriting criteria of LendingClub.com, and 0 otherwise.
2. purpose: The purpose of the loan (takes values "creditcard", "debtconsolidation", "educational", "majorpurchase", "smallbusiness", and "all_other").
3. int.rate: The interest rate of the loan, as a proportion (a rate of 11% would be stored as 0.11). Borrowers judged by LendingClub.com to be more risky are assigned higher interest rates.
4. installment: The monthly installments owed by the borrower if the loan is funded.
5. log.annual.inc: The natural log of the self-reported annual income of the borrower.
6. dti: The debt-to-income ratio of the borrower (amount of debt divided by annual income).
7. fico: The FICO credit score of the borrower.
8. days.with.cr.line: The number of days the borrower has had a credit line.
9. revol.bal: The borrower's revolving balance (amount unpaid at the end of the credit card billing cycle).
10. revol.util: The borrower's revolving line utilization rate (the amount of the credit line used relative to total credit available).
11. inq.last.6mths: The borrower's number of inquiries by creditors in the last 6 months.
12. delinq.2yrs: The number of times the borrower had been 30+ days past due on a payment in the past 2 years.
13. pub.rec: The borrower's number of derogatory public records (bankruptcy filings, tax liens, or judgments).

The project is divided into following sections
* Machine Learning Pipeline that pre-processes, normalizes and trains model to be used for classifying loans 
* Web application that takes above-mentioned parameters as input and predicts whether loan will be paid in full or not
 
## Getting Started <a name='GetStarted'></a>
### Dependencies <a name='Dependencies'></a>
Following packages were used in this project
* numpy
* sklearn
* seaborn
* matplotlib
* joblib
* flask

### Installation <a name='Installation'></a>
* Clone this repository by executing `git clone https://github.com/sumitkumar-00/loan-payoff-predictor`
* Install required packages by executing `pipenv install` in the project's root directory

### Executing program <a name='Execution'></a>
1. Run following commands in project's root directory   
   * To execute ML pipeline `pipenv run python model/classifier.py data/loan_data.csv`
   * To run web app execute `pipenv run python run.py` from app's directory
2. Go to http://127.0.0.1:3201 to check out the app 

## Authors <a name='Authors'></a>
. [Sumit Kumar](https://github.com/sumitkumar-00)
## License <a name='License'></a>
Feel free to make changes
## Acknowledgement <a name='Acknowledgement'></a>
I would like to thank Kaggle making this data available
## Screenshots <a name='Screenshots'></a>         




