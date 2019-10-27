
# Wats

How can we explain discrepancies to answer what the programmer might be asking:

###Show code clone (not just near-miss)

Show a code clone pair which **does** have a perfect or close to perfect score for the test case.

This is like trying to show user another clone that does produce the output:

- Is there another R snippet I could've used to produce the Python output?

**Challenge:** The operation that did produce the output might be performing unrelated operation(s)
- One way we can deal with that is to also calculate edit distance to get the more syntactically closer clone

**Challenge:** There are no other clones that have a good enough score


### Show an example test case dataframe (input) and highlight those cells/rows/cols that is the root cause.

This is like trying to explain the similarity score:

- Why is the score lower/higher than you expected?

Would be simple to automate generation of explanation if based on row/col difference.

For e.g. R's `head` has different number of rows printed by default btw Python/R: 

**Message:** R output *added 1 more row* when given *df*

How do you know default? Accepts params but aren't supplied any.


For e.g. R's `[` pads NA rows: 

**Message:** R output *added x rows* when given *df*

**Challenge:** How do you explain the why?

One solution would be a template based approach: discrepancy type => message (with slots)

For e.g. we can tell if the output discrepancy is a `NA` issue due to padding or due to comparison of `NA`

For padding, R adds rows where all cols are `NA` **including** the row index

For filter issue, it's when the values for the columns is `NA`

**TODO:** Find more examples that are more complex

Off-by-one:

`df.head()`	vs `head(df)` 0.9

`df.iloc[:7]` vs `slice(df, 1:8)` 0.938

`df.iloc[:7]` vs `head(df)` 0.914


NA rows when row has NA for any columns used to compare (last one compared seems to cause it):

`df.query('col1 == 1 & col3 == 1')` vs `df[df$col1 == 1 & df$col3 == 1, ]` 0.9

`df[(df.col1 == 1) & (df.col3 == 1)]` vs `df[df$col1 == 1 & df$col3 == 1, ]` 0.9


NA padding

`df.iloc[:8]` vs `df[1:8, ]` 0.65

as opposed to

`df.iloc[:8]` vs `slice(df, 1:8)` 1

More NA padding

`df.iloc[0:5, 0:3]` vs `df[1:5, 1:3]` 0.7

`df.iloc[:7]` vs `df[1:7, ]` 0.671





