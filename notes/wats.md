
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

**Message':** Python prints *5 rows by default* whereas R *prints 6 rows* by default.

Similarly, `NA` is not removed by default for calculating mean in R but Python does so:

**Message:** R output was *NA*. Make sure to remove `NA`s before calculating mean by setting `na.rm` parameter to `TRUE`

A more subtle example is R's `arrange` and `order` preserves row order but Python's `sort_values` does not.

`sort_values` has a parameter which uses quicksort algorithm to sort which is not stable:

> kind : {‘quicksort’, ‘mergesort’, ‘heapsort’}, default ‘quicksort’ Choice of sorting algorithm. See also ndarray.np.sort for more information. mergesort is the only stable algorithm. 

Discussion [here](https://stackoverflow.com/questions/19580900/how-is-pandas-deciding-order-in-a-sort-when-there-is-a-tie)

If no default params are being set, it could lead to these subtle discrepancies bc Pandas
chose a different default. Copout solution for now:

**Message:** R output *contains x rows* with *y cells* that are different when given *df*. Consider checking default parameters for both languages.

If row indices are checked and compared between the two it's possible to tell user that it's because of the default value of `kind`.

**Message:** R output *contains x rows* with *y cells* that are different when given *df*. R preserves order of row indices when sorting whereas Python does not preserve order by default. The `kind` parameter of Python must be set to 'mergesort' to preserve order of row indices.

### Use in/out highlight but also use a criteria to deliver message:

For e.g. R's `[` pads NA rows: 

**Message:** R output *added x rows* when given *df*

**Challenge:** How do you explain the why?

One solution would be a template based approach: discrepancy type => message (with slots)

For e.g. we can tell if the output discrepancy is a `NA` issue due to padding or due to comparison of `NA`

For padding, R adds rows where all cols are `NA` **including** the row index

For filter issue, it's when the values for the columns is `NA`

**Message':** R output *added x rows* when given *df* because `[` sliced for more rows than the dataframe has. R fills the gaps with `NA`s.

# Wats

**TODO:** Find more examples that are more complex

Off-by-one:

`df.head()`	vs `head(df)` 0.9

`df.iloc[:7]` vs `slice(df, 1:8)` 0.938

`df.iloc[:7]` vs `head(df)` 0.914

NA rows when row has NA for any columns used to compare (last one compared seems to cause it):

`df.query('col1 == 1 & col3 == 1')` vs `df[df$col1 == 1 & df$col3 == 1, ]` 0.9

`df[(df.col1 == 1) & (df.col3 == 1)]` vs `df[df$col1 == 1 & df$col3 == 1, ]` 0.9

NA padding at the end of dataframe

`df.iloc[:8]` vs `df[1:8, ]` 0.65

as opposed to

`df.iloc[:8]` vs `slice(df, 1:8)` 1

More NA padding

`df.iloc[0:5, 0:3]` vs `df[1:5, 1:3]` 0.7

`df.iloc[:7]` vs `df[1:7, ]` 0.671

Sorting using default expression in Pandas vs R causes discrepancy because Pandas uses quicksort:

`df.sort_values('col1', ascending=False)` vs `arrange(df, desc(col1))` 0.97

`df.sort_values('col1', ascending=False)` vs `df[order(-df$col1), ]` 0.97

To preserve rows like R does by default, you would need to specify the `kind` param and choose a stable
algorithm like mergesort:

`df.sort_values('col1', ascending=False, kind='mergesort')` vs `arrange(df, desc(col1))` 1.0

Another example of a default parameter causing a discrepancy, R by default does not remove `NA`
before calculating the mean of a column: 

`df.col1.mean()`vs `mean(df$col1)` 0.352

as opposed to

`df.col1.mean()` vs `mean(df$col1, na.rm=TRUE)`	1



