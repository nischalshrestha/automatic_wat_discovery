
# Wats

How can we explain discrepancies to answer what the programmer might be asking:

### Show code clone (not just near-miss)

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

It could be possible to know inconsistency in default behavior when function accepts params but aren't supplied any.

For e.g. R's `arrange` and `order` preserves row order but Python's `sort_values` does not.

`sort_values` has a parameter which uses quicksort algorithm to sort which is not stable:

> kind : {‘quicksort’, ‘mergesort’, ‘heapsort’}, default ‘quicksort’ Choice of sorting algorithm. See also ndarray.np.sort for more information. mergesort is the only stable algorithm. 

If no default params are being set, it could lead to these subtle discrepancies bc Pandas
chose a different default. Copout solution for now:

**Message:** R output *contains x rows* with *y cells* that are different when given *df*. Consider checking default parameters for both languages.

### Use in/out highlight but also use a criteria to deliver message:

For e.g. R's `[` pads NA rows: 

**Message:** R output *added x rows* when given *df*

**Challenge:** How do you explain the why?

One solution would be a template based approach: discrepancy type => message (with slots)

For e.g. we can tell if the output discrepancy is a `NA` issue due to padding or due to comparison of `NA`

For padding, R adds rows where all cols are `NA` **including** the row index

For filter issue, it's when the values for the columns is `NA`

# Wats

**TODO:** Find more examples that are more complex

Off-by-one:

`df.head()`	vs `head(df)` 0.9

`df.iloc[:7]` vs `slice(df, 1:8)` 0.938

`df.iloc[:7]` vs `head(df)` 0.914

NA rows when row has NA for any columns used to compare (last one compared seems to cause it):

`df.query('col1 == 1 & col3 == 1')` vs `df[df$col1 == 1 & df$col3 == 1, ]` 0.9

`df[(df.col1 == 1) & (df.col3 == 1)]` vs `df[df$col1 == 1 & df$col3 == 1, ]` 0.9

Sorting using default expression in Pandas vs R causes discrepancy because Pandas uses quicksort:

`df.sort_values('col1', ascending=False)` vs `arrange(df, desc(col1))` 0.599

`df.sort_values('col1', ascending=False)` vs `df[order(-df$col1), ]` 0.599

To preserve rows like R does by default, you would need to specify the `kind` param and choose a stable
algorithm like mergesort:

`df.sort_values('col1', ascending=False, kind='mergesort')`

NA padding

`df.iloc[:8]` vs `df[1:8, ]` 0.65

as opposed to

`df.iloc[:8]` vs `slice(df, 1:8)` 1

More NA padding

`df.iloc[0:5, 0:3]` vs `df[1:5, 1:3]` 0.7

`df.iloc[:7]` vs `df[1:7, ]` 0.671





