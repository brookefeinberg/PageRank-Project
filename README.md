# PageRank-Project

In this project, you will create a simple search engine for the website <https://www.lawfareblog.com>.
This website provides legal analysis on US national security issues.
You will use pagerank to return only the most important results from this website in your search engine.

**Due date:** Sunday, 22 September at midnight

**Late Policy:** You lose $2^{(i-1)}$ points, where i is the number of days late.

<!--
**Computation:**
This project has low computational requirements.
You should be able to complete it on your own laptops.
-->

**Collaboration Policy:**
Do whatever will help you learn,
but be an adult.
You may talk to other students and use Google/ChatGPT.
Recall that you will have an in-person oral exam on this material and the exam is worth many more points.
The main purpose of this project is to help prepare you for the exam.

## Background

**Data:**

The `data` folder contains two files that store example "web graphs".
The file `small.csv.gz` contains the example graph from the *Deeper Inside Pagerank* paper.
This is a small graph, so we can manually inspect the contents of this file with the following command:
```
$ zcat data/small.csv.gz
source,target
1,2
1,3
3,1
3,2
3,5
4,5
4,6
5,6
5,4
6,4
```

> **Recall:**
> The `cat` terminal command outputs the contents of a file to stdout, and the `zcat` command first decompressed a gzipped file and then outputs the decompressed contents.
>
> In python, we can use the built-in `gzip` module to access gzipped files.
> The following python code is equivalent to the bash code above:
>
> ```
> >>> import gzip
> >>> fin = gzip.open('data/small.csv.gz', mode='rt')
> >>> print(fin.read())
> source,target
> 1,2
> 1,3
> 3,1
> 3,2
> 3,5
> 4,5
> 4,6
> 5,6
> 5,4
> 6,4
> ```
>
> There are many terminal commands throughout these instructions.
> If you haven't used the terminal before, and so these commands are unfamiliar, that's okay.
> I'd be happy to explain them in office hours,
> or there are many tutors in the QCL available who can help.
> (There are no tutors for this class specifically, but anyone who has taken CSCI046 or CSCI133 with me will be able to help with the terminal.)
>
> Furthermore, you don't "need" to understand the terminal commands in detail,
> since you are not required to run these commands or to create your own.
> The important part is to understand the English language description of what the commands are doing,
> and to understand that this is just how I computed what the English language text is describing.

As you can see, the graph is stored as a CSV file.
The first line is a header,
and each subsequent line stores a single edge in the graph.
The first column contains the source node of the edge and the second column the target node.
The file is assumed to be sorted alphabetically.

The second data file `lawfareblog.csv.gz` contains the link structure for the lawfare blog.
Let's take a look at the first 10 of these lines:
```
$ zcat data/lawfareblog.csv.gz | head
source,target
www.lawfareblog.com/,www.lawfareblog.com/topic/interrogation
www.lawfareblog.com/,www.lawfareblog.com/upcoming-events
www.lawfareblog.com/,www.lawfareblog.com/
www.lawfareblog.com/,www.lawfareblog.com/our-comments-policy
www.lawfareblog.com/,www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
www.lawfareblog.com/,www.lawfareblog.com/topic/lawfare-research-paper-series
www.lawfareblog.com/,www.lawfareblog.com/topic/book-reviews
www.lawfareblog.com/,www.lawfareblog.com/documents-related-mueller-investigation
www.lawfareblog.com/,www.lawfareblog.com/topic/international-law-loac
```
You can see that in this file, the node names are URLs.
Semantically, each line corresponds to an HTML `<a>` tag that is contained in the source webpage and links to the target webpage.

We can use the following command to count the total number of links in the file:
```
$ zcat data/lawfareblog.csv.gz | wc -l
1610789
```
Since every link corresponds to a non-zero entry in the $P$ matrix,
this is also the value of $\text{nnz}(P)$.
(Technically, we should subtract 1 from this value since the `wc -l` command also counts the header line, not just the data lines.)

To get the dimensions of $P$, we need to count the total number of nodes in the graph.
The following command achieves this by: decompressing the file, extracting the first column, removing all duplicate lines, then counting the results.
```
$ zcat data/lawfareblog.csv.gz | cut -f1 -d, | uniq | wc -l
25761
```
This matrix is large enough that computing matrix products for dense matrices takes several minutes on a single CPU.
Fortunately, however, the matrix is very sparse.
The following python code computes the fraction of entries in the matrix with non-zero values:
```
>>> 1610788 / (25760**2)
0.0024274297384360172
```
Thus, by using sparse matrix operations, we will be able to speed up the code significantly.

**Code:**

The `pagerank.py` file contains code for loading the graph CSV files and searching through their nodes for key phrases.
For example, you can perform a search for all nodes (i.e. urls) that mention the string `corona` with the following command:
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --search_query=corona
```

> **NOTE:**
> It will take about 10 seconds to load and parse the data files.
> All the other computation happens essentially instantly.

Currently, the pagerank of the nodes is not currently being calculated correctly, and so the webpages are returned in an arbitrary order.
Your task in this assignment will be to fix these calculations in order to have the most important results (i.e. highest pagerank results) returned first.

## Task 1: the power method

Implement the `WebGraph.power_method` function in `pagerank.py` for computing the pagerank vector by fixing the [`FIXME: Task 1` annotation](https://github.com/mikeizbicki/cmc-csci145-math166/blob/81ed5d2b75f5bc23b8de93805c29321ab431ed9b/topic01_computation_pagerank/project/pagerank.py#L144).

> **NOTE:**
> The power method is the only data mining algorithm you will implement in class.
> You are implementing it because there are no standard library implementations available.
> Why?
> 1. The runtime is heavily dependent on the data structures used to store the graph data.
>    Different applications will need to use different data structures.
> 1. It is "trivial" to implement.
>    My solution to this homework is <10 lines of code.

**Part 1:**

To check that your implementation is working,
you should run the program on the `data/small.csv.gz` graph.
For my implementation, I get the following output.
```
$ python3 pagerank.py --data=data/small.csv.gz --verbose
DEBUG:root:computing indices
DEBUG:root:computing values
DEBUG:root:i=0 residual=2.5629e-01
DEBUG:root:i=1 residual=1.1841e-01
DEBUG:root:i=2 residual=7.0701e-02
DEBUG:root:i=3 residual=3.1815e-02
DEBUG:root:i=4 residual=2.0497e-02
DEBUG:root:i=5 residual=1.0108e-02
DEBUG:root:i=6 residual=6.3716e-03
DEBUG:root:i=7 residual=3.4228e-03
DEBUG:root:i=8 residual=2.0879e-03
DEBUG:root:i=9 residual=1.1750e-03
DEBUG:root:i=10 residual=7.0131e-04
DEBUG:root:i=11 residual=4.0321e-04
DEBUG:root:i=12 residual=2.3800e-04
DEBUG:root:i=13 residual=1.3812e-04
DEBUG:root:i=14 residual=8.1083e-05
DEBUG:root:i=15 residual=4.7251e-05
DEBUG:root:i=16 residual=2.7704e-05
DEBUG:root:i=17 residual=1.6164e-05
DEBUG:root:i=18 residual=9.4778e-06
DEBUG:root:i=19 residual=5.5066e-06
DEBUG:root:i=20 residual=3.2042e-06
DEBUG:root:i=21 residual=1.8612e-06
DEBUG:root:i=22 residual=1.1283e-06
DEBUG:root:i=23 residual=6.1907e-07
INFO:root:rank=0 pagerank=6.6270e-01 url=4
INFO:root:rank=1 pagerank=5.2179e-01 url=6
INFO:root:rank=2 pagerank=4.1434e-01 url=5
INFO:root:rank=3 pagerank=2.3175e-01 url=2
INFO:root:rank=4 pagerank=1.8590e-01 url=3
INFO:root:rank=5 pagerank=1.6917e-01 url=1
```
Yours likely won't be identical (due to minor implementation details and weird floating point issues), but it should be similar.
In particular, the ranking of the nodes/urls should be the same order.

> **NOTE:**
> The `--verbose` flag causes all of the lines beginning with `DEBUG` to be printed.
> By default, only lines beginning with `INFO` are printed.

> **NOTE:**
> There are no automated test cases to pass for this assignment.
> Test cases for algorithms involving floating point computations are hard to write and understand.
> Minor-seeming implementations details can have large impacts on the final result.
> These software engineering issues are beyond the scope of this class.
>
> Instructions for how I will grade your homework are contained in the [submission section](#submission) at the end of this document.

**Part 2:**

The `pagerank.py` file has an option `--search_query`, which takes a string as a parameter.
If this argument is used, then the program returns all nodes that match the query string sorted according to their pagerank.
Essentially, this gives us the most important pages related to our query.

Again, you may not get the exact same results as me,
but you should get similar results to the examples I've shown below.
Verify that you do in fact get similar results.

```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='corona'
INFO:root:rank=0 pagerank=1.0038e-03 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
INFO:root:rank=1 pagerank=8.9224e-04 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
INFO:root:rank=2 pagerank=7.0390e-04 url=www.lawfareblog.com/britains-coronavirus-response
INFO:root:rank=3 pagerank=6.9153e-04 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
INFO:root:rank=4 pagerank=6.7041e-04 url=www.lawfareblog.com/israeli-emergency-regulations-location-tracking-coronavirus-carriers
INFO:root:rank=5 pagerank=6.6256e-04 url=www.lawfareblog.com/why-congress-conducting-business-usual-face-coronavirus
INFO:root:rank=6 pagerank=6.5046e-04 url=www.lawfareblog.com/congressional-homeland-security-committees-seek-ways-support-state-federal-responses-coronavirus
INFO:root:rank=7 pagerank=6.3620e-04 url=www.lawfareblog.com/paper-hearing-experts-debate-digital-contact-tracing-and-coronavirus-privacy-concerns
INFO:root:rank=8 pagerank=6.1248e-04 url=www.lawfareblog.com/house-subcommittee-voices-concerns-over-us-management-coronavirus
INFO:root:rank=9 pagerank=6.0187e-04 url=www.lawfareblog.com/livestream-house-oversight-committee-holds-hearing-government-coronavirus-response

$ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='trump'
INFO:root:rank=0 pagerank=5.7826e-03 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
INFO:root:rank=1 pagerank=5.2338e-03 url=www.lawfareblog.com/document-trump-revokes-obama-executive-order-counterterrorism-strike-casualty-reporting
INFO:root:rank=2 pagerank=5.1297e-03 url=www.lawfareblog.com/trump-administrations-worrying-new-policy-israeli-settlements
INFO:root:rank=3 pagerank=4.6599e-03 url=www.lawfareblog.com/dc-circuit-overrules-district-courts-due-process-ruling-qasim-v-trump
INFO:root:rank=4 pagerank=4.5934e-03 url=www.lawfareblog.com/donald-trump-and-politically-weaponized-executive-branch
INFO:root:rank=5 pagerank=4.3071e-03 url=www.lawfareblog.com/how-trumps-approach-middle-east-ignores-past-future-and-human-condition
INFO:root:rank=6 pagerank=4.0935e-03 url=www.lawfareblog.com/why-trump-cant-buy-greenland
INFO:root:rank=7 pagerank=3.7591e-03 url=www.lawfareblog.com/oral-argument-summary-qassim-v-trump
INFO:root:rank=8 pagerank=3.4509e-03 url=www.lawfareblog.com/dc-circuit-court-denies-trump-rehearing-mazars-case
INFO:root:rank=9 pagerank=3.4484e-03 url=www.lawfareblog.com/second-circuit-rules-mazars-must-hand-over-trump-tax-returns-new-york-prosecutors

$ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='iran'
INFO:root:rank=0 pagerank=4.5746e-03 url=www.lawfareblog.com/praise-presidents-iran-tweets
INFO:root:rank=1 pagerank=4.4174e-03 url=www.lawfareblog.com/how-us-iran-tensions-could-disrupt-iraqs-fragile-peace
INFO:root:rank=2 pagerank=2.6928e-03 url=www.lawfareblog.com/cyber-command-operational-update-clarifying-june-2019-iran-operation
INFO:root:rank=3 pagerank=1.9391e-03 url=www.lawfareblog.com/aborted-iran-strike-fine-line-between-necessity-and-revenge
INFO:root:rank=4 pagerank=1.5452e-03 url=www.lawfareblog.com/parsing-state-departments-letter-use-force-against-iran
INFO:root:rank=5 pagerank=1.5357e-03 url=www.lawfareblog.com/iranian-hostage-crisis-and-its-effect-american-politics
INFO:root:rank=6 pagerank=1.5258e-03 url=www.lawfareblog.com/announcing-united-states-and-use-force-against-iran-new-lawfare-e-book
INFO:root:rank=7 pagerank=1.4221e-03 url=www.lawfareblog.com/us-names-iranian-revolutionary-guard-terrorist-organization-and-sanctions-international-criminal
INFO:root:rank=8 pagerank=1.1788e-03 url=www.lawfareblog.com/iran-shoots-down-us-drone-domestic-and-international-legal-implications
INFO:root:rank=9 pagerank=1.1463e-03 url=www.lawfareblog.com/israel-iran-syria-clash-and-law-use-force
```

**Part 3:**

The webgraph of lawfareblog.com (i.e. the $P$ matrix) naturally contains a lot of structure.
For example, essentially all pages on the domain have links to the root page <https://lawfareblog.com/> and other "non-article" pages like <https://www.lawfareblog.com/topics> and <https://www.lawfareblog.com/subscribe-lawfare>.
These pages therefore have a large pagerank.
We can get a list of the pages with the largest pagerank by running

```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz
INFO:root:rank=0 pagerank=2.8741e-01 url=www.lawfareblog.com/lawfare-job-board
INFO:root:rank=1 pagerank=2.8741e-01 url=www.lawfareblog.com/masthead
INFO:root:rank=2 pagerank=2.8741e-01 url=www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
INFO:root:rank=3 pagerank=2.8741e-01 url=www.lawfareblog.com/documents-related-mueller-investigation
INFO:root:rank=4 pagerank=2.8741e-01 url=www.lawfareblog.com/topics
INFO:root:rank=5 pagerank=2.8741e-01 url=www.lawfareblog.com/about-lawfare-brief-history-term-and-site
INFO:root:rank=6 pagerank=2.8741e-01 url=www.lawfareblog.com/snowden-revelations
INFO:root:rank=7 pagerank=2.8741e-01 url=www.lawfareblog.com/support-lawfare
INFO:root:rank=8 pagerank=2.8741e-01 url=www.lawfareblog.com/upcoming-events
INFO:root:rank=9 pagerank=2.8741e-01 url=www.lawfareblog.com/our-comments-policy
```

Most of these pages are not very interesting, however, because they are not articles,
and usually when we are performing a web search, we only want articles.

This raises the question: How can we find the most important articles filtering out the non-article pages?
The answer is to modify the $P$ matrix by removing all links to non-article pages.

This raises another question: How do we know if a link is a non-article page?
Unfortunately, this is a hard question to answer with 100% accuracy,
but there are many methods that get us most of the way there.
One easy to implement method is to compute what's called the "in-link ratio" of each node (i.e. the total number of edges with the node as a target divided by the total number of nodes),
and then remove nodes from the search results with too-high of a ratio.
The intuition is that non-article pages often appear in the menu of a webpage, and so have links from almost all of the other webpages;
but article-webpages are unlikely to appear on a menu and so will only have a small number of links from other webpages.
The `--filter_ratio` parameter causes the code to remove all pages that have an in-link ratio larger than the provided value.

Using this option, we can estimate the most important articles on the domain with the following command:
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2
INFO:root:rank=0 pagerank=3.4696e-01 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
INFO:root:rank=1 pagerank=2.9521e-01 url=www.lawfareblog.com/livestream-nov-21-impeachment-hearings-0
INFO:root:rank=2 pagerank=2.9040e-01 url=www.lawfareblog.com/opening-statement-david-holmes
INFO:root:rank=3 pagerank=1.5179e-01 url=www.lawfareblog.com/lawfare-podcast-ben-nimmo-whack-mole-game-disinformation
INFO:root:rank=4 pagerank=1.5099e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1963
INFO:root:rank=5 pagerank=1.5099e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1964
INFO:root:rank=6 pagerank=1.5071e-01 url=www.lawfareblog.com/lawfare-podcast-week-was-impeachment
INFO:root:rank=7 pagerank=1.4957e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1962
INFO:root:rank=8 pagerank=1.4367e-01 url=www.lawfareblog.com/cyberlaw-podcast-mistrusting-google
INFO:root:rank=9 pagerank=1.4240e-01 url=www.lawfareblog.com/lawfare-podcast-bonus-edition-gordon-sondland-vs-committee-no-bull
```
Notice that the urls in this list look much more like articles than the urls in the previous list.

When Google calculates their $P$ matrix for the web,
they use a similar (but much more complicated) process to modify the $P$ matrix in order to reduce spam results.
The exact formula they use is a jealously guarded secret that they update continuously.

In the case above, notice that we have accidentally removed the blog's most popular article (<https://www.lawfareblog.com/snowden-revelations>).
The blog editors believed that Snowden's revelations about NSA spying are so important that they directly put a link to the article on the menu.
So every single webpage in the domain links to the Snowden article,
and our "anti-spam" `--filter-ratio` argument removed this article from the list.
In general, it is a challenging open problem to remove spam from pagerank results,
and all current solutions rely on careful human tuning and still have lots of false positives and false negatives.

**Part 4:**

Recall from the reading that the runtime of pagerank depends heavily on the eigengap of the $\bar{\bar P}$ matrix,
and that this eigengap is bounded by the alpha parameter.

Run the following four commands:
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose 
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --alpha=0.99999
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2 --alpha=0.99999
```
You should notice that the last command takes considerably more iterations to compute the pagerank vector.
(My code takes 685 iterations for this call, and about 10 iterations for all the others.)

This raises the question: Why does the second command (with the `--alpha` option but without the `--filter_ratio`) option not take a long time to run?
The answer is that the $P$ graph for <https://www.lawfareblog.com> naturally has a large eigengap and so is fast to compute for all alpha values,
but the modified graph does not have a large eigengap and so requires a small alpha for fast convergence.

Changing the value of alpha also gives us very different pagerank rankings.
For example, 
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2
INFO:root:rank=0 pagerank=3.4696e-01 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
INFO:root:rank=1 pagerank=2.9521e-01 url=www.lawfareblog.com/livestream-nov-21-impeachment-hearings-0
INFO:root:rank=2 pagerank=2.9040e-01 url=www.lawfareblog.com/opening-statement-david-holmes
INFO:root:rank=3 pagerank=1.5179e-01 url=www.lawfareblog.com/lawfare-podcast-ben-nimmo-whack-mole-game-disinformation
INFO:root:rank=4 pagerank=1.5099e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1963
INFO:root:rank=5 pagerank=1.5099e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1964
INFO:root:rank=6 pagerank=1.5071e-01 url=www.lawfareblog.com/lawfare-podcast-week-was-impeachment
INFO:root:rank=7 pagerank=1.4957e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1962
INFO:root:rank=8 pagerank=1.4367e-01 url=www.lawfareblog.com/cyberlaw-podcast-mistrusting-google
INFO:root:rank=9 pagerank=1.4240e-01 url=www.lawfareblog.com/lawfare-podcast-bonus-edition-gordon-sondland-vs-committee-no-bull

$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --alpha=0.99999
INFO:root:rank=0 pagerank=7.0149e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
INFO:root:rank=1 pagerank=7.0149e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
INFO:root:rank=2 pagerank=1.0552e-01 url=www.lawfareblog.com/cost-using-zero-days
INFO:root:rank=3 pagerank=3.1755e-02 url=www.lawfareblog.com/lawfare-podcast-former-congressman-brian-baird-and-daniel-schuman-how-congress-can-continue-function
INFO:root:rank=4 pagerank=2.2040e-02 url=www.lawfareblog.com/events
INFO:root:rank=5 pagerank=1.6027e-02 url=www.lawfareblog.com/water-wars-increased-us-focus-indo-pacific
INFO:root:rank=6 pagerank=1.6026e-02 url=www.lawfareblog.com/water-wars-drill-maybe-drill
INFO:root:rank=7 pagerank=1.6023e-02 url=www.lawfareblog.com/water-wars-disjointed-operations-south-china-sea
INFO:root:rank=8 pagerank=1.6020e-02 url=www.lawfareblog.com/water-wars-song-oil-and-fire
INFO:root:rank=9 pagerank=1.6020e-02 url=www.lawfareblog.com/water-wars-sinking-feeling-philippine-china-relations
```

Which of these rankings is better is entirely subjective,
and the only way to know if you have the "best" alpha for your application is to try several variations and see what is best.

> **NOTE:**
> It should be "obvious" to you that large alpha values imply that the structure of the webgraph has more influence on the final result,
> and small alpha values ignore the structure of the webgraph.
> Recall that the word "obvious" means that it follows directly from the definition,
> but you may still need to sit and meditate on the definition for a long period of time.

If large alphas are good for your application, you can see that there is a trade-off between quality answers and algorithmic runtime.
We'll be exploring this trade-off more formally in class over the rest of the semester.

## Task 2: the personalization vector

The most interesting applications of pagerank involve the personalization vector.
Implement the `WebGraph.make_personalization_vector` function so that it outputs a personalization vector tuned for the input query.
The pseudocode for the function is:
```
for each index in the personalization vector:
    get the url for the index (see the _index_to_url function)
    check if the url satisfies the input query (see the url_satisfies_query function)
    if so, set the corresponding index to one
normalize the vector
```

**Part 1:**

The command line argument `--personalization_vector_query` will use the function you created above to augment your search with a custom personalization vector.
If you've implemented the function correctly,
you should get results similar to:
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='corona'
INFO:root:rank=0 pagerank=6.3127e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
INFO:root:rank=1 pagerank=6.3124e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
INFO:root:rank=2 pagerank=1.5947e-01 url=www.lawfareblog.com/chinatalk-how-party-takes-its-propaganda-global
INFO:root:rank=3 pagerank=1.2209e-01 url=www.lawfareblog.com/brexit-not-immune-coronavirus
INFO:root:rank=4 pagerank=1.2209e-01 url=www.lawfareblog.com/rational-security-my-corona-edition
INFO:root:rank=5 pagerank=9.3360e-02 url=www.lawfareblog.com/trump-cant-reopen-country-over-state-objections
INFO:root:rank=6 pagerank=9.1920e-02 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
INFO:root:rank=7 pagerank=9.1920e-02 url=www.lawfareblog.com/britains-coronavirus-response
INFO:root:rank=8 pagerank=7.7770e-02 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
INFO:root:rank=9 pagerank=7.2888e-02 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
```

Notice that these results are significantly different than when using the `--search_query` option:
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --search_query='corona'
INFO:root:rank=0 pagerank=8.1320e-03 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
INFO:root:rank=1 pagerank=7.7908e-03 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
INFO:root:rank=2 pagerank=5.2262e-03 url=www.lawfareblog.com/livestream-house-oversight-committee-holds-hearing-government-coronavirus-response
INFO:root:rank=3 pagerank=3.9584e-03 url=www.lawfareblog.com/britains-coronavirus-response
INFO:root:rank=4 pagerank=3.8114e-03 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
INFO:root:rank=5 pagerank=3.3973e-03 url=www.lawfareblog.com/paper-hearing-experts-debate-digital-contact-tracing-and-coronavirus-privacy-concerns
INFO:root:rank=6 pagerank=3.3633e-03 url=www.lawfareblog.com/cyberlaw-podcast-how-israel-fighting-coronavirus
INFO:root:rank=7 pagerank=3.3557e-03 url=www.lawfareblog.com/israeli-emergency-regulations-location-tracking-coronavirus-carriers
INFO:root:rank=8 pagerank=3.2160e-03 url=www.lawfareblog.com/congress-needs-coronavirus-failsafe-its-too-late
INFO:root:rank=9 pagerank=3.1036e-03 url=www.lawfareblog.com/why-congress-conducting-business-usual-face-coronavirus
```

Which results are better?
Again, that depends on what you mean by "better."
With the `--personalization_vector_query` option,
a webpage is important only if other coronavirus webpages also think it's important;
with the `--search_query` option,
a webpage is important if any other webpage thinks it's important.
You'll notice that in the later example, many of the webpages are about Congressional proceedings related to the coronavirus.
From a strictly coronavirus perspective, these are not very important webpages.
But in the broader context of national security, these are very important webpages.

Google engineers spend TONs of time fine-tuning their pagerank personalization vectors to remove spam webpages.
Exactly how they do this is another one of their secrets that they don't publicly talk about.

**Part 2:**

Another use of the `--personalization_vector_query` option is that we can find out what webpages are related to the coronavirus but don't directly mention the coronavirus.
This can be used to map out what types of topics are similar to the coronavirus.

For example, the following query ranks all webpages by their `corona` importance,
but removes webpages mentioning `corona` from the results.
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='corona' --search_query='-corona'
INFO:root:rank=0 pagerank=6.3127e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
INFO:root:rank=1 pagerank=6.3124e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
INFO:root:rank=2 pagerank=1.5947e-01 url=www.lawfareblog.com/chinatalk-how-party-takes-its-propaganda-global
INFO:root:rank=3 pagerank=9.3360e-02 url=www.lawfareblog.com/trump-cant-reopen-country-over-state-objections
INFO:root:rank=4 pagerank=7.0277e-02 url=www.lawfareblog.com/fault-lines-foreign-policy-quarantined
INFO:root:rank=5 pagerank=6.9713e-02 url=www.lawfareblog.com/lawfare-podcast-mom-and-dad-talk-clinical-trials-pandemic
INFO:root:rank=6 pagerank=6.4944e-02 url=www.lawfareblog.com/limits-world-health-organization
INFO:root:rank=7 pagerank=5.9492e-02 url=www.lawfareblog.com/chinatalk-dispatches-shanghai-beijing-and-hong-kong
INFO:root:rank=8 pagerank=5.1245e-02 url=www.lawfareblog.com/us-moves-dismiss-case-against-company-linked-ira-troll-farm
INFO:root:rank=9 pagerank=5.1245e-02 url=www.lawfareblog.com/livestream-house-armed-services-holds-hearing-national-security-challenges-north-and-south-america
```
You can see that there are many urls about concepts that are obviously related like "covid", "clinical trials", and "quarantine",
but this algorithm also finds articles about Chinese propaganda and Trump's policy decisions.
Both of these articles are highly relevant to coronavirus discussions,
but a simple keyword search for corona or related terms would not find these articles.
The vast majority of industry data mining work is finding clever uses of standard algorithms.

<!--
**Part 3:**

Select another topic related to national security.
You should experiment with a national security topic other than the coronavirus.
For example, find out what articles are important to the `iran` topic but do not contain the word `iran`.
Your goal should be to discover what topics that www.lawfareblog.com considers to be related to the national security topic you choose.
-->

## Submission

1. Create a new repo on github (not a fork of this repo).
    Ensure that all of the project files are copied from this folder into your new repo.

1. As you complete the tasks above:
    Run the corresponding commands below, and paste their output into the code blocks.
    Please ensure correct markdown formatting.
   
   Task 1, part 1:
   ```
   $ python3 pagerank.py --data=data/small.csv.gz --verbose
   DEBUG:root:computing indices
    DEBUG:root:computing values
    DEBUG:root:i=0 residual=0.3470110595226288
    DEBUG:root:i=1 residual=0.1935427337884903
    DEBUG:root:i=2 residual=0.08822998404502869
    DEBUG:root:i=3 residual=0.032490864396095276
    DEBUG:root:i=4 residual=0.017331792041659355
    DEBUG:root:i=5 residual=0.006763693410903215
    DEBUG:root:i=6 residual=0.0034524539951235056
    DEBUG:root:i=7 residual=0.0013982367236167192
    DEBUG:root:i=8 residual=0.0006716411444358528
    DEBUG:root:i=9 residual=0.00027746849809773266
    DEBUG:root:i=10 residual=0.00012771245383191854
    DEBUG:root:i=11 residual=5.33431775693316e-05
    DEBUG:root:i=12 residual=2.3841603251639754e-05
    DEBUG:root:i=13 residual=9.984948519559111e-06
    DEBUG:root:i=14 residual=4.366116627352312e-06
    DEBUG:root:i=15 residual=1.8554338794274372e-06
    DEBUG:root:i=16 residual=8.431345008830249e-07
    INFO:root:rank=0 pagerank=6.0257e-01 url=4
    INFO:root:rank=1 pagerank=4.6414e-01 url=6
    INFO:root:rank=2 pagerank=3.4544e-01 url=5
    INFO:root:rank=3 pagerank=1.2732e-01 url=2
    INFO:root:rank=4 pagerank=9.9210e-02 url=3
    INFO:root:rank=5 pagerank=8.9347e-02 url=1
   ```

   Task 1, part 2:
   ```
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='corona'
   INFO:root:rank=0 pagerank=4.5865e-03 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
   INFO:root:rank=1 pagerank=4.0464e-03 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
   INFO:root:rank=2 pagerank=2.6118e-03 url=www.lawfareblog.com/britains-coronavirus-response
   INFO:root:rank=3 pagerank=2.5392e-03 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
   INFO:root:rank=4 pagerank=2.3560e-03 url=www.lawfareblog.com/israeli-emergency-regulations-location-tracking-coronavirus-carriers
   INFO:root:rank=5 pagerank=2.2897e-03 url=www.lawfareblog.com/why-congress-conducting-business-usual-face-coronavirus
   INFO:root:rank=6 pagerank=2.2729e-03 url=www.lawfareblog.com/livestream-house-oversight-committee-holds-hearing-government-coronavirus-response
   INFO:root:rank=7 pagerank=2.2522e-03 url=www.lawfareblog.com/congressional-homeland-security-committees-seek-ways-support-state-federal-responses-coronavirus
   INFO:root:rank=8 pagerank=2.1880e-03 url=www.lawfareblog.com/paper-hearing-experts-debate-digital-contact-tracing-and-coronavirus-privacy-concerns
   INFO:root:rank=9 pagerank=2.0341e-03 url=www.lawfareblog.com/cyberlaw-podcast-how-israel-fighting-coronavirus

   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='trump'
   INFO:root:rank=0 pagerank=6.6250e-02 url=www.lawfareblog.com/donald-trump-and-politically-weaponized-executive-branch
   INFO:root:rank=1 pagerank=6.0200e-02 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
   INFO:root:rank=2 pagerank=3.4972e-02 url=www.lawfareblog.com/trump-administrations-worrying-new-policy-israeli-settlements
   INFO:root:rank=3 pagerank=3.2196e-02 url=www.lawfareblog.com/document-trump-revokes-obama-executive-order-counterterrorism-strike-casualty-reporting
   INFO:root:rank=4 pagerank=3.0974e-02 url=www.lawfareblog.com/dc-circuit-overrules-district-courts-due-process-ruling-qasim-v-trump
   INFO:root:rank=5 pagerank=2.8463e-02 url=www.lawfareblog.com/how-trumps-approach-middle-east-ignores-past-future-and-human-condition
   INFO:root:rank=6 pagerank=2.5255e-02 url=www.lawfareblog.com/why-trump-cant-buy-greenland
   INFO:root:rank=7 pagerank=2.2459e-02 url=www.lawfareblog.com/oral-argument-summary-qassim-v-trump
   INFO:root:rank=8 pagerank=2.1464e-02 url=www.lawfareblog.com/dc-circuit-court-denies-trump-rehearing-mazars-case
   INFO:root:rank=9 pagerank=2.1105e-02 url=www.lawfareblog.com/second-circuit-rules-mazars-must-hand-over-trump-tax-returns-new-york-prosecutors

   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='iran'
   INFO:root:rank=0 pagerank=6.6138e-02 url=www.lawfareblog.com/praise-presidents-iran-tweets
   INFO:root:rank=1 pagerank=2.9202e-02 url=www.lawfareblog.com/how-us-iran-tensions-could-disrupt-iraqs-fragile-peace
   INFO:root:rank=2 pagerank=1.7711e-02 url=www.lawfareblog.com/cyber-command-operational-update-clarifying-june-2019-iran-operation
   INFO:root:rank=3 pagerank=1.4606e-02 url=www.lawfareblog.com/aborted-iran-strike-fine-line-between-necessity-and-revenge
   INFO:root:rank=4 pagerank=8.4519e-03 url=www.lawfareblog.com/iranian-hostage-crisis-and-its-effect-american-politics
   INFO:root:rank=5 pagerank=8.3997e-03 url=www.lawfareblog.com/parsing-state-departments-letter-use-force-against-iran
   INFO:root:rank=6 pagerank=8.2589e-03 url=www.lawfareblog.com/announcing-united-states-and-use-force-against-iran-new-lawfare-e-book
   INFO:root:rank=7 pagerank=8.0568e-03 url=www.lawfareblog.com/trump-moves-cut-irans-oil-revenues-whats-his-endgame
   INFO:root:rank=8 pagerank=7.1946e-03 url=www.lawfareblog.com/us-names-iranian-revolutionary-guard-terrorist-organization-and-sanctions-international-criminal
   INFO:root:rank=9 pagerank=5.9410e-03 url=www.lawfareblog.com/iran-shoots-down-us-drone-domestic-and-international-legal-implications
   ```

   Task 1, part 3:
   ```
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz
    INFO:root:rank=0 pagerank=8.4165e+00 url=www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
    INFO:root:rank=1 pagerank=8.4165e+00 url=www.lawfareblog.com/lawfare-job-board
    INFO:root:rank=2 pagerank=8.4165e+00 url=www.lawfareblog.com/documents-related-mueller-investigation
    INFO:root:rank=3 pagerank=8.4165e+00 url=www.lawfareblog.com/litigation-documents-resources-related-travel-ban
    INFO:root:rank=4 pagerank=8.4165e+00 url=www.lawfareblog.com/subscribe-lawfare
    INFO:root:rank=5 pagerank=8.4165e+00 url=www.lawfareblog.com/masthead
    INFO:root:rank=6 pagerank=8.4165e+00 url=www.lawfareblog.com/topics
    INFO:root:rank=7 pagerank=8.4165e+00 url=www.lawfareblog.com/our-comments-policy
    INFO:root:rank=8 pagerank=8.4165e+00 url=www.lawfareblog.com/upcoming-events
    INFO:root:rank=9 pagerank=8.4165e+00 url=www.lawfareblog.com/about-lawfare-brief-history-term-and-site

   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2
    INFO:root:rank=0 pagerank=4.2777e+00 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
    INFO:root:rank=1 pagerank=2.7719e+00 url=www.lawfareblog.com/livestream-nov-21-impeachment-hearings-0
    INFO:root:rank=2 pagerank=2.7535e+00 url=www.lawfareblog.com/opening-statement-david-holmes
    INFO:root:rank=3 pagerank=1.8722e+00 url=www.lawfareblog.com/senate-examines-threats-homeland
    INFO:root:rank=4 pagerank=1.7419e+00 url=www.lawfareblog.com/what-make-first-day-impeachment-hearings
    INFO:root:rank=5 pagerank=1.7412e+00 url=www.lawfareblog.com/livestream-house-armed-services-committee-hearing-f-35-program
    INFO:root:rank=6 pagerank=1.7349e+00 url=www.lawfareblog.com/whats-house-resolution-impeachment
    INFO:root:rank=7 pagerank=1.6385e+00 url=www.lawfareblog.com/congress-us-policy-toward-syria-and-turkey-overview-recent-hearings
    INFO:root:rank=8 pagerank=1.5598e+00 url=www.lawfareblog.com/summary-david-holmess-deposition-testimony
    INFO:root:rank=9 pagerank=9.1273e-01 url=www.lawfareblog.com/events
   ```

   Task 1, part 4:
   ```
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose
    DEBUG:root:computing indices
    DEBUG:root:computing values
    DEBUG:root:i=0 residual=20.521549224853516
    DEBUG:root:i=1 residual=6.110842227935791
    DEBUG:root:i=2 residual=1.9216899871826172
    DEBUG:root:i=3 residual=0.5883616805076599
    DEBUG:root:i=4 residual=0.17544226348400116
    DEBUG:root:i=5 residual=0.05154351517558098
    DEBUG:root:i=6 residual=0.014947004616260529
    DEBUG:root:i=7 residual=0.004287370480597019
    DEBUG:root:i=8 residual=0.00116316182538867
    DEBUG:root:i=9 residual=0.00026119101676158607
    DEBUG:root:i=10 residual=3.8402664358727634e-05
    DEBUG:root:i=11 residual=7.030449341982603e-05
    DEBUG:root:i=12 residual=7.607047882629558e-05
    DEBUG:root:i=13 residual=6.939918239368126e-05
    DEBUG:root:i=14 residual=5.94819757679943e-05
    DEBUG:root:i=15 residual=4.956904012942687e-05
    DEBUG:root:i=16 residual=3.9655315049458295e-05
    DEBUG:root:i=17 residual=3.63497129001189e-05
    DEBUG:root:i=18 residual=3.304426354588941e-05
    DEBUG:root:i=19 residual=2.6437322958372533e-05
    DEBUG:root:i=20 residual=1.9829183656838723e-05
    DEBUG:root:i=21 residual=1.9825167328235693e-05
    DEBUG:root:i=22 residual=1.6523486920050345e-05
    DEBUG:root:i=23 residual=1.6522120859008282e-05
    DEBUG:root:i=24 residual=9.917439456330612e-06
    DEBUG:root:i=25 residual=1.3215803846833296e-05
    DEBUG:root:i=26 residual=6.611609478568425e-06
    DEBUG:root:i=27 residual=9.911661436490249e-06
    DEBUG:root:i=28 residual=6.611585831706179e-06
    DEBUG:root:i=29 residual=3.30589318764396e-06
    DEBUG:root:i=30 residual=6.192430390683512e-08
    INFO:root:rank=0 pagerank=8.4165e+00 url=www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
    INFO:root:rank=1 pagerank=8.4165e+00 url=www.lawfareblog.com/lawfare-job-board
    INFO:root:rank=2 pagerank=8.4165e+00 url=www.lawfareblog.com/documents-related-mueller-investigation
    INFO:root:rank=3 pagerank=8.4165e+00 url=www.lawfareblog.com/litigation-documents-resources-related-travel-ban
    INFO:root:rank=4 pagerank=8.4165e+00 url=www.lawfareblog.com/subscribe-lawfare
    INFO:root:rank=5 pagerank=8.4165e+00 url=www.lawfareblog.com/masthead
    INFO:root:rank=6 pagerank=8.4165e+00 url=www.lawfareblog.com/topics
    INFO:root:rank=7 pagerank=8.4165e+00 url=www.lawfareblog.com/our-comments-policy
    INFO:root:rank=8 pagerank=8.4165e+00 url=www.lawfareblog.com/upcoming-events
    INFO:root:rank=9 pagerank=8.4165e+00 url=www.lawfareblog.com/about-lawfare-brief-history-term-and-site
   
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --alpha=0.99999
    DEBUG:root:computing indices
    DEBUG:root:computing values
    DEBUG:root:i=0 residual=24.140165328979492
    DEBUG:root:i=1 residual=8.458995819091797
    DEBUG:root:i=2 residual=3.128697395324707
    DEBUG:root:i=3 residual=1.124985694885254
    DEBUG:root:i=4 residual=0.3952835202217102
    DEBUG:root:i=5 residual=0.13689588010311127
    DEBUG:root:i=6 residual=0.04697730019688606
    DEBUG:root:i=7 residual=0.015959959477186203
    DEBUG:root:i=8 residual=0.005272224545478821
    DEBUG:root:i=9 residual=0.0015919979196041822
    DEBUG:root:i=10 residual=0.0003603620862122625
    DEBUG:root:i=11 residual=0.00015777892258483917
    DEBUG:root:i=12 residual=0.000262540765106678
    DEBUG:root:i=13 residual=0.00029754923889413476
    DEBUG:root:i=14 residual=0.0003007303166668862
    DEBUG:root:i=15 residual=0.000310624047415331
    DEBUG:root:i=16 residual=0.0003238390199840069
    DEBUG:root:i=17 residual=0.0003172420838382095
    DEBUG:root:i=18 residual=0.00031723815482109785
    DEBUG:root:i=19 residual=0.00031723809661343694
    DEBUG:root:i=20 residual=0.0003172381257172674
    DEBUG:root:i=21 residual=0.0003172381257172674
    DEBUG:root:i=22 residual=0.00031723949359729886
    DEBUG:root:i=23 residual=0.0003172381839249283
    DEBUG:root:i=24 residual=0.000317238038405776
    DEBUG:root:i=25 residual=0.0003139354521408677
    DEBUG:root:i=26 residual=0.000317236699629575
    DEBUG:root:i=27 residual=0.00031723809661343694
    DEBUG:root:i=28 residual=0.00031723809661343694
    DEBUG:root:i=29 residual=0.0003172381839249283
    DEBUG:root:i=30 residual=0.0003172381839249283
    DEBUG:root:i=31 residual=0.0003172380675096065
    DEBUG:root:i=32 residual=0.0003172381839249283
    DEBUG:root:i=33 residual=0.00031723809661343694
    DEBUG:root:i=34 residual=0.0003172381257172674
    DEBUG:root:i=35 residual=0.00031723943538963795
    DEBUG:root:i=36 residual=0.0003172381257172674
    DEBUG:root:i=37 residual=0.00031723809661343694
    DEBUG:root:i=38 residual=0.00031723821302875876
    DEBUG:root:i=39 residual=0.0003172381839249283
    DEBUG:root:i=40 residual=0.000317238038405776
    DEBUG:root:i=41 residual=0.0003172380675096065
    DEBUG:root:i=42 residual=0.00031723809661343694
    DEBUG:root:i=43 residual=0.0003172381839249283
    DEBUG:root:i=44 residual=0.00031723949359729886
    DEBUG:root:i=45 residual=0.0003139355394523591
    DEBUG:root:i=46 residual=0.00031723675783723593
    DEBUG:root:i=47 residual=0.000317238038405776
    DEBUG:root:i=48 residual=0.0003205407992936671
    DEBUG:root:i=49 residual=0.00032054222538135946
    DEBUG:root:i=50 residual=0.00031723958090879023
    DEBUG:root:i=51 residual=0.0003172395227011293
    DEBUG:root:i=52 residual=0.0003172381257172674
    DEBUG:root:i=53 residual=0.0003172381257172674
    DEBUG:root:i=54 residual=0.00031393542303703725
    DEBUG:root:i=55 residual=0.00031723675783723593
    DEBUG:root:i=56 residual=0.0003172380675096065
    DEBUG:root:i=57 residual=0.0003172381257172674
    DEBUG:root:i=58 residual=0.000317238038405776
    DEBUG:root:i=59 residual=0.0003172381839249283
    DEBUG:root:i=60 residual=0.0003172380675096065
    DEBUG:root:i=61 residual=0.00031723815482109785
    DEBUG:root:i=62 residual=0.00031723809661343694
    DEBUG:root:i=63 residual=0.00031723943538963795
    DEBUG:root:i=64 residual=0.0003172381257172674
    DEBUG:root:i=65 residual=0.00031723815482109785
    DEBUG:root:i=66 residual=0.00031723809661343694
    DEBUG:root:i=67 residual=0.0003172382421325892
    DEBUG:root:i=68 residual=0.0003172382421325892
    DEBUG:root:i=69 residual=0.000317238038405776
    DEBUG:root:i=70 residual=0.0003172380675096065
    DEBUG:root:i=71 residual=0.00031723809661343694
    DEBUG:root:i=72 residual=0.00031393548124469817
    DEBUG:root:i=73 residual=0.00031723672873340547
    DEBUG:root:i=74 residual=0.0003172380675096065
    DEBUG:root:i=75 residual=0.00031723809661343694
    DEBUG:root:i=76 residual=0.00031723949359729886
    DEBUG:root:i=77 residual=0.00031723821302875876
    DEBUG:root:i=78 residual=0.000317238038405776
    DEBUG:root:i=79 residual=0.00032054088660515845
    DEBUG:root:i=80 residual=0.0003172394062858075
    DEBUG:root:i=81 residual=0.00031723821302875876
    DEBUG:root:i=82 residual=0.0003172394644934684
    DEBUG:root:i=83 residual=0.00031723815482109785
    DEBUG:root:i=84 residual=0.0003172381257172674
    DEBUG:root:i=85 residual=0.00031723815482109785
    DEBUG:root:i=86 residual=0.000317238038405776
    DEBUG:root:i=87 residual=0.00031723809661343694
    DEBUG:root:i=88 residual=0.0003172381839249283
    DEBUG:root:i=89 residual=0.0003172381257172674
    DEBUG:root:i=90 residual=0.0003172381257172674
    DEBUG:root:i=91 residual=0.0003172380675096065
    DEBUG:root:i=92 residual=0.00031723815482109785
    DEBUG:root:i=93 residual=0.0003172395227011293
    DEBUG:root:i=94 residual=0.00031723815482109785
    DEBUG:root:i=95 residual=0.00031393548124469817
    DEBUG:root:i=96 residual=0.00031723672873340547
    DEBUG:root:i=97 residual=0.0003172379801981151
    DEBUG:root:i=98 residual=0.00031723809661343694
    DEBUG:root:i=99 residual=0.0003172381257172674
    DEBUG:root:i=100 residual=0.00031723809661343694
    DEBUG:root:i=101 residual=0.00031723815482109785
    DEBUG:root:i=102 residual=0.000317238038405776
    DEBUG:root:i=103 residual=0.0003172381257172674
    DEBUG:root:i=104 residual=0.0003205407992936671
    DEBUG:root:i=105 residual=0.0003172409487888217
    DEBUG:root:i=106 residual=0.00032054088660515845
    DEBUG:root:i=107 residual=0.0003172395227011293
    DEBUG:root:i=108 residual=0.0003172381839249283
    DEBUG:root:i=109 residual=0.0003139368782285601
    DEBUG:root:i=110 residual=0.00031723533174954355
    DEBUG:root:i=111 residual=0.0003205407701898366
    DEBUG:root:i=112 residual=0.0003172409487888217
    DEBUG:root:i=113 residual=0.0003172381257172674
    DEBUG:root:i=114 residual=0.00031723809661343694
    DEBUG:root:i=115 residual=0.0003172381257172674
    DEBUG:root:i=116 residual=0.0003172381257172674
    DEBUG:root:i=117 residual=0.00031723821302875876
    DEBUG:root:i=118 residual=0.0003172381257172674
    DEBUG:root:i=119 residual=0.0003172394644934684
    DEBUG:root:i=120 residual=0.0003172381257172674
    DEBUG:root:i=121 residual=0.0003172381257172674
    DEBUG:root:i=122 residual=0.00031723815482109785
    DEBUG:root:i=123 residual=0.00031723809661343694
    DEBUG:root:i=124 residual=0.0003172381257172674
    DEBUG:root:i=125 residual=0.00031723809661343694
    DEBUG:root:i=126 residual=0.0003139354521408677
    DEBUG:root:i=127 residual=0.00031723672873340547
    DEBUG:root:i=128 residual=0.0003172381257172674
    DEBUG:root:i=129 residual=0.0003172381257172674
    DEBUG:root:i=130 residual=0.000317238038405776
    DEBUG:root:i=131 residual=0.0003172394644934684
    DEBUG:root:i=132 residual=0.00031723815482109785
    DEBUG:root:i=133 residual=0.0003172381839249283
    DEBUG:root:i=134 residual=0.00031723809661343694
    DEBUG:root:i=135 residual=0.0003172381839249283
    DEBUG:root:i=136 residual=0.00031723800930194557
    DEBUG:root:i=137 residual=0.0003172381839249283
    DEBUG:root:i=138 residual=0.000317238038405776
    DEBUG:root:i=139 residual=0.0003172381257172674
    DEBUG:root:i=140 residual=0.0003172381839249283
    DEBUG:root:i=141 residual=0.0003172395227011293
    DEBUG:root:i=142 residual=0.00031723821302875876
    DEBUG:root:i=143 residual=0.000317238038405776
    DEBUG:root:i=144 residual=0.00031723815482109785
    DEBUG:root:i=145 residual=0.00031723815482109785
    DEBUG:root:i=146 residual=0.0003172381257172674
    DEBUG:root:i=147 residual=0.0003172381257172674
    DEBUG:root:i=148 residual=0.0003172381257172674
    DEBUG:root:i=149 residual=0.0003172381257172674
    DEBUG:root:i=150 residual=0.0003172380675096065
    DEBUG:root:i=151 residual=0.0003238449280615896
    DEBUG:root:i=152 residual=0.0003205435932613909
    DEBUG:root:i=153 residual=0.0003172409487888217
    DEBUG:root:i=154 residual=0.0003172381257172674
    DEBUG:root:i=155 residual=0.0003172381839249283
    DEBUG:root:i=156 residual=0.00031723815482109785
    DEBUG:root:i=157 residual=0.000317238038405776
    DEBUG:root:i=158 residual=0.00031723809661343694
    DEBUG:root:i=159 residual=0.0003139353939332068
    DEBUG:root:i=160 residual=0.00031723672873340547
    DEBUG:root:i=161 residual=0.0003172380675096065
    DEBUG:root:i=162 residual=0.0003172381839249283
    DEBUG:root:i=163 residual=0.00031393542303703725
    DEBUG:root:i=164 residual=0.00031393414246849716
    DEBUG:root:i=165 residual=0.0003205394314136356
    DEBUG:root:i=166 residual=0.00031723943538963795
    DEBUG:root:i=167 residual=0.00031723958090879023
    DEBUG:root:i=168 residual=0.0003172381257172674
    DEBUG:root:i=169 residual=0.0003172380675096065
    DEBUG:root:i=170 residual=0.0003172381257172674
    DEBUG:root:i=171 residual=0.0003172381257172674
    DEBUG:root:i=172 residual=0.00031723809661343694
    DEBUG:root:i=173 residual=0.0003172381257172674
    DEBUG:root:i=174 residual=0.0003172381257172674
    DEBUG:root:i=175 residual=0.0003172380675096065
    DEBUG:root:i=176 residual=0.0003172394644934684
    DEBUG:root:i=177 residual=0.00031723815482109785
    DEBUG:root:i=178 residual=0.0003172381839249283
    DEBUG:root:i=179 residual=0.0003172381257172674
    DEBUG:root:i=180 residual=0.000317238038405776
    DEBUG:root:i=181 residual=0.0003172380675096065
    DEBUG:root:i=182 residual=0.00032054082839749753
    DEBUG:root:i=183 residual=0.00032054356415756047
    DEBUG:root:i=184 residual=0.00031723963911645114
    DEBUG:root:i=185 residual=0.0003172382421325892
    DEBUG:root:i=186 residual=0.0003172380675096065
    DEBUG:root:i=187 residual=0.0003172381257172674
    DEBUG:root:i=188 residual=0.0003172380675096065
    DEBUG:root:i=189 residual=0.0003139355394523591
    DEBUG:root:i=190 residual=0.00031723675783723593
    DEBUG:root:i=191 residual=0.000317238038405776
    DEBUG:root:i=192 residual=0.0003172380675096065
    DEBUG:root:i=193 residual=0.0003139367909170687
    DEBUG:root:i=194 residual=0.0003139326872769743
    DEBUG:root:i=195 residual=0.0003172366414219141
    DEBUG:root:i=196 residual=0.00031723809661343694
    DEBUG:root:i=197 residual=0.000317238038405776
    DEBUG:root:i=198 residual=0.00032054082839749753
    DEBUG:root:i=199 residual=0.0003172408614773303
    DEBUG:root:i=200 residual=0.0003172382421325892
    DEBUG:root:i=201 residual=0.00031723809661343694
    DEBUG:root:i=202 residual=0.0003139354521408677
    DEBUG:root:i=203 residual=0.00031723672873340547
    DEBUG:root:i=204 residual=0.0003172381257172674
    DEBUG:root:i=205 residual=0.00031723815482109785
    DEBUG:root:i=206 residual=0.0003205407701898366
    DEBUG:root:i=207 residual=0.0003172410069964826
    DEBUG:root:i=208 residual=0.000317238038405776
    DEBUG:root:i=209 residual=0.0003172381257172674
    DEBUG:root:i=210 residual=0.0003139354521408677
    DEBUG:root:i=211 residual=0.00032053946051746607
    DEBUG:root:i=212 residual=0.0003172394644934684
    DEBUG:root:i=213 residual=0.0003172382421325892
    DEBUG:root:i=214 residual=0.00031393542303703725
    DEBUG:root:i=215 residual=0.00032053940230980515
    DEBUG:root:i=216 residual=0.00031724091968499124
    DEBUG:root:i=217 residual=0.0003172381257172674
    DEBUG:root:i=218 residual=0.00031723815482109785
    DEBUG:root:i=219 residual=0.0003139354521408677
    DEBUG:root:i=220 residual=0.00031723667052574456
    DEBUG:root:i=221 residual=0.0003172380675096065
    DEBUG:root:i=222 residual=0.0003172381257172674
    DEBUG:root:i=223 residual=0.00031723815482109785
    DEBUG:root:i=224 residual=0.00031723815482109785
    DEBUG:root:i=225 residual=0.0003172380675096065
    DEBUG:root:i=226 residual=0.0003172380675096065
    DEBUG:root:i=227 residual=0.00031723958090879023
    DEBUG:root:i=228 residual=0.0003172381257172674
    DEBUG:root:i=229 residual=0.0003172381257172674
    DEBUG:root:i=230 residual=0.00031723809661343694
    DEBUG:root:i=231 residual=0.00031723809661343694
    DEBUG:root:i=232 residual=0.000317238038405776
    DEBUG:root:i=233 residual=0.00031723815482109785
    DEBUG:root:i=234 residual=0.0003172381257172674
    DEBUG:root:i=235 residual=0.00031723809661343694
    DEBUG:root:i=236 residual=0.00031393548124469817
    DEBUG:root:i=237 residual=0.0003205393732059747
    DEBUG:root:i=238 residual=0.0003172410069964826
    DEBUG:root:i=239 residual=0.0003172381257172674
    DEBUG:root:i=240 residual=0.0003172381839249283
    DEBUG:root:i=241 residual=0.0003172381257172674
    DEBUG:root:i=242 residual=0.0003172380675096065
    DEBUG:root:i=243 residual=0.00031723815482109785
    DEBUG:root:i=244 residual=0.0003172381257172674
    DEBUG:root:i=245 residual=0.00031723815482109785
    DEBUG:root:i=246 residual=0.00031723809661343694
    DEBUG:root:i=247 residual=0.0003172394644934684
    DEBUG:root:i=248 residual=0.0003172381257172674
    DEBUG:root:i=249 residual=0.0003172381257172674
    DEBUG:root:i=250 residual=0.00031723809661343694
    DEBUG:root:i=251 residual=0.0003172381839249283
    DEBUG:root:i=252 residual=0.0003172381257172674
    DEBUG:root:i=253 residual=0.000317238038405776
    DEBUG:root:i=254 residual=0.00031723815482109785
    DEBUG:root:i=255 residual=0.000317238038405776
    DEBUG:root:i=256 residual=0.00031723958090879023
    DEBUG:root:i=257 residual=0.0003172381257172674
    DEBUG:root:i=258 residual=0.0003172381257172674
    DEBUG:root:i=259 residual=0.000317238038405776
    DEBUG:root:i=260 residual=0.0003172381257172674
    DEBUG:root:i=261 residual=0.00031393542303703725
    DEBUG:root:i=262 residual=0.0003172367869410664
    DEBUG:root:i=263 residual=0.0003172381839249283
    DEBUG:root:i=264 residual=0.0003139353357255459
    DEBUG:root:i=265 residual=0.0003139340551570058
    DEBUG:root:i=266 residual=0.0003172366414219141
    DEBUG:root:i=267 residual=0.0003172381257172674
    DEBUG:root:i=268 residual=0.00031393542303703725
    DEBUG:root:i=269 residual=0.00032053946051746607
    DEBUG:root:i=270 residual=0.0003172394062858075
    DEBUG:root:i=271 residual=0.000320540857501328
    DEBUG:root:i=272 residual=0.00031724091968499124
    DEBUG:root:i=273 residual=0.0003139355103485286
    DEBUG:root:i=274 residual=0.0003172368451487273
    DEBUG:root:i=275 residual=0.0003172379801981151
    DEBUG:root:i=276 residual=0.00031723809661343694
    DEBUG:root:i=277 residual=0.0003172381257172674
    DEBUG:root:i=278 residual=0.0003172381257172674
    DEBUG:root:i=279 residual=0.0003172380675096065
    DEBUG:root:i=280 residual=0.00031723815482109785
    DEBUG:root:i=281 residual=0.0003139353066217154
    DEBUG:root:i=282 residual=0.00031723681604489684
    DEBUG:root:i=283 residual=0.000317238038405776
    DEBUG:root:i=284 residual=0.0003172381257172674
    DEBUG:root:i=285 residual=0.00031723815482109785
    DEBUG:root:i=286 residual=0.0003172381839249283
    DEBUG:root:i=287 residual=0.0003172395518049598
    DEBUG:root:i=288 residual=0.0003172380675096065
    DEBUG:root:i=289 residual=0.00031723815482109785
    DEBUG:root:i=290 residual=0.00031723809661343694
    DEBUG:root:i=291 residual=0.00031723815482109785
    DEBUG:root:i=292 residual=0.0003172380675096065
    DEBUG:root:i=293 residual=0.0003139354521408677
    DEBUG:root:i=294 residual=0.0003040260635316372
    DEBUG:root:i=295 residual=0.00030732303275726736
    DEBUG:root:i=296 residual=0.00031392977689392865
    DEBUG:root:i=297 residual=0.0003139337641187012
    DEBUG:root:i=298 residual=0.0003106312651652843
    DEBUG:root:i=299 residual=0.000313931202981621
    DEBUG:root:i=300 residual=0.0003040259762201458
    DEBUG:root:i=301 residual=0.0003139284090138972
    DEBUG:root:i=302 residual=0.00031393388053402305
    DEBUG:root:i=303 residual=0.00031393388053402305
    DEBUG:root:i=304 residual=0.00031723527354188263
    DEBUG:root:i=305 residual=0.00031723809661343694
    DEBUG:root:i=306 residual=0.0003172380675096065
    DEBUG:root:i=307 residual=0.00032054228358902037
    DEBUG:root:i=308 residual=0.00031723949359729886
    DEBUG:root:i=309 residual=0.0003172381257172674
    DEBUG:root:i=310 residual=0.00031723815482109785
    DEBUG:root:i=311 residual=0.00031723815482109785
    DEBUG:root:i=312 residual=0.0003139354521408677
    DEBUG:root:i=313 residual=0.0003172366414219141
    DEBUG:root:i=314 residual=0.00031723815482109785
    DEBUG:root:i=315 residual=0.0003172380675096065
    DEBUG:root:i=316 residual=0.00032054082839749753
    DEBUG:root:i=317 residual=0.00031724091968499124
    DEBUG:root:i=318 residual=0.00031723821302875876
    DEBUG:root:i=319 residual=0.0003172381257172674
    DEBUG:root:i=320 residual=0.00031393542303703725
    DEBUG:root:i=321 residual=0.00031723675783723593
    DEBUG:root:i=322 residual=0.00031723809661343694
    DEBUG:root:i=323 residual=0.0003172381257172674
    DEBUG:root:i=324 residual=0.000317238038405776
    DEBUG:root:i=325 residual=0.0003172381257172674
    DEBUG:root:i=326 residual=0.0003172394644934684
    DEBUG:root:i=327 residual=0.00031723821302875876
    DEBUG:root:i=328 residual=0.00031393548124469817
    DEBUG:root:i=329 residual=0.00031723667052574456
    DEBUG:root:i=330 residual=0.0003172380675096065
    DEBUG:root:i=331 residual=0.00031723809661343694
    DEBUG:root:i=332 residual=0.00032054082839749753
    DEBUG:root:i=333 residual=0.000320542196277529
    DEBUG:root:i=334 residual=0.00031724103610031307
    DEBUG:root:i=335 residual=0.0003172381257172674
    DEBUG:root:i=336 residual=0.0003172381257172674
    DEBUG:root:i=337 residual=0.0003172381257172674
    DEBUG:root:i=338 residual=0.0003172382421325892
    DEBUG:root:i=339 residual=0.0003139354521408677
    DEBUG:root:i=340 residual=0.00031723667052574456
    DEBUG:root:i=341 residual=0.0003172380675096065
    DEBUG:root:i=342 residual=0.0003172380675096065
    DEBUG:root:i=343 residual=0.0003172381257172674
    DEBUG:root:i=344 residual=0.0003172381257172674
    DEBUG:root:i=345 residual=0.0003172381257172674
    DEBUG:root:i=346 residual=0.0003172395518049598
    DEBUG:root:i=347 residual=0.0003172380675096065
    DEBUG:root:i=348 residual=0.00031723815482109785
    DEBUG:root:i=349 residual=0.000317238038405776
    DEBUG:root:i=350 residual=0.00031723815482109785
    DEBUG:root:i=351 residual=0.00031723809661343694
    DEBUG:root:i=352 residual=0.00031723809661343694
    DEBUG:root:i=353 residual=0.0003172381257172674
    DEBUG:root:i=354 residual=0.0003172381257172674
    DEBUG:root:i=355 residual=0.00031723949359729886
    DEBUG:root:i=356 residual=0.0003139355103485286
    DEBUG:root:i=357 residual=0.00031723675783723593
    DEBUG:root:i=358 residual=0.000317238038405776
    DEBUG:root:i=359 residual=0.0003172381257172674
    DEBUG:root:i=360 residual=0.00031393548124469817
    DEBUG:root:i=361 residual=0.0003139341133646667
    DEBUG:root:i=362 residual=0.00031393259996548295
    DEBUG:root:i=363 residual=0.00031723661231808364
    DEBUG:root:i=364 residual=0.0003205407701898366
    DEBUG:root:i=365 residual=0.0003205435932613909
    DEBUG:root:i=366 residual=0.0003172395518049598
    DEBUG:root:i=367 residual=0.00031723821302875876
    DEBUG:root:i=368 residual=0.0003139354521408677
    DEBUG:root:i=369 residual=0.0003172366414219141
    DEBUG:root:i=370 residual=0.0003172381257172674
    DEBUG:root:i=371 residual=0.0003172380675096065
    DEBUG:root:i=372 residual=0.0003172381839249283
    DEBUG:root:i=373 residual=0.0003172394644934684
    DEBUG:root:i=374 residual=0.00031723809661343694
    DEBUG:root:i=375 residual=0.00031723809661343694
    DEBUG:root:i=376 residual=0.00031723815482109785
    DEBUG:root:i=377 residual=0.0003172380675096065
    DEBUG:root:i=378 residual=0.00031723815482109785
    DEBUG:root:i=379 residual=0.0003172381257172674
    DEBUG:root:i=380 residual=0.0003172381257172674
    DEBUG:root:i=381 residual=0.0003139354521408677
    DEBUG:root:i=382 residual=0.000317236699629575
    DEBUG:root:i=383 residual=0.0003172381257172674
    DEBUG:root:i=384 residual=0.0003172381257172674
    DEBUG:root:i=385 residual=0.00031723815482109785
    DEBUG:root:i=386 residual=0.00031723943538963795
    DEBUG:root:i=387 residual=0.0003172381839249283
    DEBUG:root:i=388 residual=0.0003172381257172674
    DEBUG:root:i=389 residual=0.0003139354521408677
    DEBUG:root:i=390 residual=0.0003040261217392981
    DEBUG:root:i=391 residual=0.0003106255899183452
    DEBUG:root:i=392 residual=0.00031062844209372997
    DEBUG:root:i=393 residual=0.0003139324835501611
    DEBUG:root:i=394 residual=0.0003106298390775919
    DEBUG:root:i=395 residual=0.0003139325708616525
    DEBUG:root:i=396 residual=0.00030732856248505414
    DEBUG:root:i=397 residual=0.0003073244006372988
    DEBUG:root:i=398 residual=0.00031392977689392865
    DEBUG:root:i=399 residual=0.00031723655411042273
    DEBUG:root:i=400 residual=0.0003172380675096065
    DEBUG:root:i=401 residual=0.000317238038405776
    DEBUG:root:i=402 residual=0.00031723809661343694
    DEBUG:root:i=403 residual=0.00031393548124469817
    DEBUG:root:i=404 residual=0.00031723672873340547
    DEBUG:root:i=405 residual=0.000317238038405776
    DEBUG:root:i=406 residual=0.0003172381257172674
    DEBUG:root:i=407 residual=0.0003172380675096065
    DEBUG:root:i=408 residual=0.00031723809661343694
    DEBUG:root:i=409 residual=0.0003172396100126207
    DEBUG:root:i=410 residual=0.0003172381839249283
    DEBUG:root:i=411 residual=0.00031723809661343694
    DEBUG:root:i=412 residual=0.000317238038405776
    DEBUG:root:i=413 residual=0.00031393542303703725
    DEBUG:root:i=414 residual=0.0003106313815806061
    DEBUG:root:i=415 residual=0.00031723390566185117
    DEBUG:root:i=416 residual=0.000317238038405776
    DEBUG:root:i=417 residual=0.00031723949359729886
    DEBUG:root:i=418 residual=0.00031393542303703725
    DEBUG:root:i=419 residual=0.00030402603442780674
    DEBUG:root:i=420 residual=0.00030732309096492827
    DEBUG:root:i=421 residual=0.00031392977689392865
    DEBUG:root:i=422 residual=0.00031393254175782204
    DEBUG:root:i=423 residual=0.00031063114874996245
    DEBUG:root:i=424 residual=0.0003139324835501611
    DEBUG:root:i=425 residual=0.0003040260635316372
    DEBUG:root:i=426 residual=0.0003073230036534369
    DEBUG:root:i=427 residual=0.00031062698690220714
    DEBUG:root:i=428 residual=0.00031393105746246874
    DEBUG:root:i=429 residual=0.00031393388053402305
    DEBUG:root:i=430 residual=0.000317236699629575
    DEBUG:root:i=431 residual=0.000317238038405776
    DEBUG:root:i=432 residual=0.0003172381257172674
    DEBUG:root:i=433 residual=0.0003172381257172674
    DEBUG:root:i=434 residual=0.0003172381257172674
    DEBUG:root:i=435 residual=0.00031723809661343694
    DEBUG:root:i=436 residual=0.00031723815482109785
    DEBUG:root:i=437 residual=0.0003172381257172674
    DEBUG:root:i=438 residual=0.0003172381257172674
    DEBUG:root:i=439 residual=0.0003172395227011293
    DEBUG:root:i=440 residual=0.0003172381839249283
    DEBUG:root:i=441 residual=0.0003139354521408677
    DEBUG:root:i=442 residual=0.00030732867890037596
    DEBUG:root:i=443 residual=0.00030732445884495974
    DEBUG:root:i=444 residual=0.0003106270160060376
    DEBUG:root:i=445 residual=0.00031393245444633067
    DEBUG:root:i=446 residual=0.0003106298390775919
    DEBUG:root:i=447 residual=0.00031393254175782204
    DEBUG:root:i=448 residual=0.0003040260053239763
    DEBUG:root:i=449 residual=0.00030732303275726736
    DEBUG:root:i=450 residual=0.0003139296604786068
    DEBUG:root:i=451 residual=0.00031393388053402305
    DEBUG:root:i=452 residual=0.00031062980997376144
    DEBUG:root:i=453 residual=0.0003139325126539916
    DEBUG:root:i=454 residual=0.0003040260635316372
    DEBUG:root:i=455 residual=0.00030732291634194553
    DEBUG:root:i=456 residual=0.00031062704510986805
    DEBUG:root:i=457 residual=0.00031062838388606906
    DEBUG:root:i=458 residual=0.00031393259996548295
    DEBUG:root:i=459 residual=0.0003040245792362839
    DEBUG:root:i=460 residual=0.00031062698690220714
    DEBUG:root:i=461 residual=0.00031062838388606906
    DEBUG:root:i=462 residual=0.0003139311447739601
    DEBUG:root:i=463 residual=0.00031063120695762336
    DEBUG:root:i=464 residual=0.0003172353026457131
    DEBUG:root:i=465 residual=0.00031723800930194557
    DEBUG:root:i=466 residual=0.0003205407992936671
    DEBUG:root:i=467 residual=0.0003172395518049598
    DEBUG:root:i=468 residual=0.00031723809661343694
    DEBUG:root:i=469 residual=0.00031723821302875876
    DEBUG:root:i=470 residual=0.00031723943538963795
    DEBUG:root:i=471 residual=0.0003172381257172674
    DEBUG:root:i=472 residual=0.0003172381257172674
    DEBUG:root:i=473 residual=0.0003172381257172674
    DEBUG:root:i=474 residual=0.0003172381257172674
    DEBUG:root:i=475 residual=0.00031393542303703725
    DEBUG:root:i=476 residual=0.00031723672873340547
    DEBUG:root:i=477 residual=0.000317238038405776
    DEBUG:root:i=478 residual=0.0003172381257172674
    DEBUG:root:i=479 residual=0.00031393548124469817
    DEBUG:root:i=480 residual=0.0003073287371080369
    DEBUG:root:i=481 residual=0.00030732431332580745
    DEBUG:root:i=482 residual=0.0003139312320854515
    DEBUG:root:i=483 residual=0.0003172352153342217
    DEBUG:root:i=484 residual=0.000317238038405776
    DEBUG:root:i=485 residual=0.00031723809661343694
    DEBUG:root:i=486 residual=0.00031393548124469817
    DEBUG:root:i=487 residual=0.00031723661231808364
    DEBUG:root:i=488 residual=0.00031723809661343694
    DEBUG:root:i=489 residual=0.00031723809661343694
    DEBUG:root:i=490 residual=0.0003172395518049598
    DEBUG:root:i=491 residual=0.0003172381257172674
    DEBUG:root:i=492 residual=0.0003172380675096065
    DEBUG:root:i=493 residual=0.0003172380675096065
    DEBUG:root:i=494 residual=0.00031723815482109785
    DEBUG:root:i=495 residual=0.00031723809661343694
    DEBUG:root:i=496 residual=0.0003172381257172674
    DEBUG:root:i=497 residual=0.00031723809661343694
    DEBUG:root:i=498 residual=0.0003172380675096065
    DEBUG:root:i=499 residual=0.0003205407992936671
    DEBUG:root:i=500 residual=0.0003172409487888217
    DEBUG:root:i=501 residual=0.0003172382421325892
    DEBUG:root:i=502 residual=0.000317238038405776
    DEBUG:root:i=503 residual=0.00031723809661343694
    DEBUG:root:i=504 residual=0.00031723809661343694
    DEBUG:root:i=505 residual=0.0003139354521408677
    DEBUG:root:i=506 residual=0.0003172367869410664
    DEBUG:root:i=507 residual=0.00031723821302875876
    DEBUG:root:i=508 residual=0.0003172381839249283
    DEBUG:root:i=509 residual=0.0003139367327094078
    DEBUG:root:i=510 residual=0.00031063001370057464
    DEBUG:root:i=511 residual=0.0003172352444380522
    DEBUG:root:i=512 residual=0.0003139353939332068
    DEBUG:root:i=513 residual=0.0003073286497965455
    DEBUG:root:i=514 residual=0.0003073244006372988
    DEBUG:root:i=515 residual=0.00031062698690220714
    DEBUG:root:i=516 residual=0.00031723518623039126
    DEBUG:root:i=517 residual=0.000317238038405776
    DEBUG:root:i=518 residual=0.00031393542303703725
    DEBUG:root:i=519 residual=0.00031063135247677565
    DEBUG:root:i=520 residual=0.00031393117387779057
    DEBUG:root:i=521 residual=0.00030732862069271505
    DEBUG:root:i=522 residual=0.0003073244006372988
    DEBUG:root:i=523 residual=0.00031393111567012966
    DEBUG:root:i=524 residual=0.0003139325126539916
    DEBUG:root:i=525 residual=0.0003172365832142532
    DEBUG:root:i=526 residual=0.000317238038405776
    DEBUG:root:i=527 residual=0.0003172381257172674
    DEBUG:root:i=528 residual=0.0003172381257172674
    DEBUG:root:i=529 residual=0.00031723809661343694
    DEBUG:root:i=530 residual=0.00031723815482109785
    DEBUG:root:i=531 residual=0.0003172394644934684
    DEBUG:root:i=532 residual=0.00031393556855618954
    DEBUG:root:i=533 residual=0.00031063129426911473
    DEBUG:root:i=534 residual=0.00031393126118928194
    DEBUG:root:i=535 residual=0.00030732856248505414
    DEBUG:root:i=536 residual=0.00030732437153346837
    DEBUG:root:i=537 residual=0.00031062704510986805
    DEBUG:root:i=538 residual=0.0003139324835501611
    DEBUG:root:i=539 residual=0.00031063129426911473
    DEBUG:root:i=540 residual=0.0003139311447739601
    DEBUG:root:i=541 residual=0.00030732862069271505
    DEBUG:root:i=542 residual=0.00030732437153346837
    DEBUG:root:i=543 residual=0.00031062698690220714
    DEBUG:root:i=544 residual=0.00031393254175782204
    DEBUG:root:i=545 residual=0.0003106312651652843
    DEBUG:root:i=546 residual=0.00031393111567012966
    DEBUG:root:i=547 residual=0.0003040260053239763
    DEBUG:root:i=548 residual=0.0003073230036534369
    DEBUG:root:i=549 residual=0.0003139297477900982
    DEBUG:root:i=550 residual=0.0003139339096378535
    DEBUG:root:i=551 residual=0.0003073285333812237
    DEBUG:root:i=552 residual=0.00030732437153346837
    DEBUG:root:i=553 residual=0.00030402158154174685
    DEBUG:root:i=554 residual=0.00031062561902217567
    DEBUG:root:i=555 residual=0.0003106284129898995
    DEBUG:root:i=556 residual=0.0003139325126539916
    DEBUG:root:i=557 residual=0.0003106312360614538
    DEBUG:root:i=558 residual=0.00031393126118928194
    DEBUG:root:i=559 residual=0.00030732856248505414
    DEBUG:root:i=560 residual=0.0003106271324213594
    DEBUG:root:i=561 residual=0.0003139324253425002
    DEBUG:root:i=562 residual=0.0003106298390775919
    DEBUG:root:i=563 residual=0.0003139325708616525
    DEBUG:root:i=564 residual=0.00030732862069271505
    DEBUG:root:i=565 residual=0.0003073244297411293
    DEBUG:root:i=566 residual=0.0003106269286945462
    DEBUG:root:i=567 residual=0.0003139324835501611
    DEBUG:root:i=568 residual=0.0003106298390775919
    DEBUG:root:i=569 residual=0.00031393259996548295
    DEBUG:root:i=570 residual=0.00030732856248505414
    DEBUG:root:i=571 residual=0.0003040217561647296
    DEBUG:root:i=572 residual=0.00031392835080623627
    DEBUG:root:i=573 residual=0.00031723655411042273
    DEBUG:root:i=574 residual=0.00031723809661343694
    DEBUG:root:i=575 residual=0.00031723809661343694
    DEBUG:root:i=576 residual=0.0003172380675096065
    DEBUG:root:i=577 residual=0.0003139353939332068
    DEBUG:root:i=578 residual=0.0003106313815806061
    DEBUG:root:i=579 residual=0.00031393126118928194
    DEBUG:root:i=580 residual=0.0003073285915888846
    DEBUG:root:i=581 residual=0.00030732445884495974
    DEBUG:root:i=582 residual=0.00031062838388606906
    DEBUG:root:i=583 residual=0.00031393111567012966
    DEBUG:root:i=584 residual=0.0003106312360614538
    DEBUG:root:i=585 residual=0.0003139312320854515
    DEBUG:root:i=586 residual=0.00030732856248505414
    DEBUG:root:i=587 residual=0.0003073244006372988
    DEBUG:root:i=588 residual=0.0003139310865662992
    DEBUG:root:i=589 residual=0.0003172351571265608
    DEBUG:root:i=590 residual=0.0003172380675096065
    DEBUG:root:i=591 residual=0.0003172380675096065
    DEBUG:root:i=592 residual=0.0003172380675096065
    DEBUG:root:i=593 residual=0.00031723949359729886
    DEBUG:root:i=594 residual=0.0003172381839249283
    DEBUG:root:i=595 residual=0.00031393542303703725
    DEBUG:root:i=596 residual=0.0003172367869410664
    DEBUG:root:i=597 residual=0.00031723795109428465
    DEBUG:root:i=598 residual=0.0003172381839249283
    DEBUG:root:i=599 residual=0.0003172380675096065
    DEBUG:root:i=600 residual=0.0003172381257172674
    DEBUG:root:i=601 residual=0.00031723809661343694
    DEBUG:root:i=602 residual=0.00031723815482109785
    DEBUG:root:i=603 residual=0.000317238038405776
    DEBUG:root:i=604 residual=0.0003172381839249283
    DEBUG:root:i=605 residual=0.0003172395227011293
    DEBUG:root:i=606 residual=0.0003172381839249283
    DEBUG:root:i=607 residual=0.0003172381257172674
    DEBUG:root:i=608 residual=0.00031393542303703725
    DEBUG:root:i=609 residual=0.0003106313233729452
    DEBUG:root:i=610 residual=0.0003139311447739601
    DEBUG:root:i=611 residual=0.0003073286497965455
    DEBUG:root:i=612 residual=0.0003073244006372988
    DEBUG:root:i=613 residual=0.00031062847119756043
    DEBUG:root:i=614 residual=0.00031393105746246874
    DEBUG:root:i=615 residual=0.0003073285333812237
    DEBUG:root:i=616 residual=0.0003040217561647296
    DEBUG:root:i=617 residual=0.000307322945445776
    DEBUG:root:i=618 residual=0.0003139296895824373
    DEBUG:root:i=619 residual=0.0003172365250065923
    DEBUG:root:i=620 residual=0.000317238038405776
    DEBUG:root:i=621 residual=0.00031723809661343694
    DEBUG:root:i=622 residual=0.00032054082839749753
    DEBUG:root:i=623 residual=0.0003205421380698681
    DEBUG:root:i=624 residual=0.0003172409487888217
    DEBUG:root:i=625 residual=0.00031393548124469817
    DEBUG:root:i=626 residual=0.00031723675783723593
    DEBUG:root:i=627 residual=0.0003172381257172674
    DEBUG:root:i=628 residual=0.00031723809661343694
    DEBUG:root:i=629 residual=0.00031723815482109785
    DEBUG:root:i=630 residual=0.0003172381257172674
    DEBUG:root:i=631 residual=0.00031723815482109785
    DEBUG:root:i=632 residual=0.0003172381257172674
    DEBUG:root:i=633 residual=0.0003172381839249283
    DEBUG:root:i=634 residual=0.0003172395227011293
    DEBUG:root:i=635 residual=0.000317238038405776
    DEBUG:root:i=636 residual=0.00031723815482109785
    DEBUG:root:i=637 residual=0.000317238038405776
    DEBUG:root:i=638 residual=0.00031723815482109785
    DEBUG:root:i=639 residual=0.00031723809661343694
    DEBUG:root:i=640 residual=0.0003172381257172674
    DEBUG:root:i=641 residual=0.0003139353357255459
    DEBUG:root:i=642 residual=0.0003172367869410664
    DEBUG:root:i=643 residual=0.0003172380675096065
    DEBUG:root:i=644 residual=0.0003172381257172674
    DEBUG:root:i=645 residual=0.00031393548124469817
    DEBUG:root:i=646 residual=0.0003106312651652843
    DEBUG:root:i=647 residual=0.00031393259996548295
    DEBUG:root:i=648 residual=0.00030732862069271505
    DEBUG:root:i=649 residual=0.0003073244006372988
    DEBUG:root:i=650 residual=0.0003139297477900982
    DEBUG:root:i=651 residual=0.00032054007169790566
    DEBUG:root:i=652 residual=0.00031393725657835603
    DEBUG:root:i=653 residual=0.0003172369033563882
    DEBUG:root:i=654 residual=0.0003172382421325892
    DEBUG:root:i=655 residual=0.00031723821302875876
    DEBUG:root:i=656 residual=0.00031723809661343694
    DEBUG:root:i=657 residual=0.0003172381257172674
    DEBUG:root:i=658 residual=0.000317238038405776
    DEBUG:root:i=659 residual=0.00031723949359729886
    DEBUG:root:i=660 residual=0.0003139354521408677
    DEBUG:root:i=661 residual=0.0003172367869410664
    DEBUG:root:i=662 residual=0.0003172381257172674
    DEBUG:root:i=663 residual=0.00031723809661343694
    DEBUG:root:i=664 residual=0.0003172380675096065
    DEBUG:root:i=665 residual=0.00031723809661343694
    DEBUG:root:i=666 residual=0.0003172381257172674
    DEBUG:root:i=667 residual=0.0003172381257172674
    DEBUG:root:i=668 residual=0.00031723821302875876
    DEBUG:root:i=669 residual=0.00031723809661343694
    DEBUG:root:i=670 residual=0.00031723809661343694
    DEBUG:root:i=671 residual=0.0003172381257172674
    DEBUG:root:i=672 residual=0.0003172395518049598
    DEBUG:root:i=673 residual=0.00031723815482109785
    DEBUG:root:i=674 residual=0.00031723809661343694
    DEBUG:root:i=675 residual=0.0003139353939332068
    DEBUG:root:i=676 residual=0.00031063129426911473
    DEBUG:root:i=677 residual=0.0003139312320854515
    DEBUG:root:i=678 residual=0.0003073285915888846
    DEBUG:root:i=679 residual=0.00030732445884495974
    DEBUG:root:i=680 residual=0.0003106283547822386
    DEBUG:root:i=681 residual=0.00031393111567012966
    DEBUG:root:i=682 residual=0.0003106312651652843
    DEBUG:root:i=683 residual=0.0003172338765580207
    DEBUG:root:i=684 residual=0.0003172380675096065
    DEBUG:root:i=685 residual=0.00031723809661343694
    DEBUG:root:i=686 residual=0.0003172394644934684
    DEBUG:root:i=687 residual=0.0003172381257172674
    DEBUG:root:i=688 residual=0.00031723815482109785
    DEBUG:root:i=689 residual=0.000317238038405776
    DEBUG:root:i=690 residual=0.00031393548124469817
    DEBUG:root:i=691 residual=0.00031063129426911473
    DEBUG:root:i=692 residual=0.0003139312320854515
    DEBUG:root:i=693 residual=0.00030732856248505414
    DEBUG:root:i=694 residual=0.00031392983510158956
    DEBUG:root:i=695 residual=0.0003172365832142532
    DEBUG:root:i=696 residual=0.00031723800930194557
    DEBUG:root:i=697 residual=0.00031393542303703725
    DEBUG:root:i=698 residual=0.00031723667052574456
    DEBUG:root:i=699 residual=0.0003172381257172674
    DEBUG:root:i=700 residual=0.0003172380675096065
    DEBUG:root:i=701 residual=0.00031723815482109785
    DEBUG:root:i=702 residual=0.00031723809661343694
    DEBUG:root:i=703 residual=0.00031723949359729886
    DEBUG:root:i=704 residual=0.00031723821302875876
    DEBUG:root:i=705 residual=0.0003172381257172674
    DEBUG:root:i=706 residual=0.00031723821302875876
    DEBUG:root:i=707 residual=0.0003139353357255459
    DEBUG:root:i=708 residual=0.00031063141068443656
    DEBUG:root:i=709 residual=0.0003139312320854515
    DEBUG:root:i=710 residual=0.00030732862069271505
    DEBUG:root:i=711 residual=0.0003073244006372988
    DEBUG:root:i=712 residual=0.00031062844209372997
    DEBUG:root:i=713 residual=0.0003172337601426989
    DEBUG:root:i=714 residual=0.0003205406537745148
    DEBUG:root:i=715 residual=0.00031723949359729886
    DEBUG:root:i=716 residual=0.00031723958090879023
    DEBUG:root:i=717 residual=0.0003172382421325892
    DEBUG:root:i=718 residual=0.000317238038405776
    DEBUG:root:i=719 residual=0.00031723809661343694
    DEBUG:root:i=720 residual=0.0003172380675096065
    DEBUG:root:i=721 residual=0.0003172381257172674
    DEBUG:root:i=722 residual=0.00031723815482109785
    DEBUG:root:i=723 residual=0.000317238038405776
    DEBUG:root:i=724 residual=0.0003172381257172674
    DEBUG:root:i=725 residual=0.0003172381257172674
    DEBUG:root:i=726 residual=0.00031723958090879023
    DEBUG:root:i=727 residual=0.0003172381257172674
    DEBUG:root:i=728 residual=0.0003139354521408677
    DEBUG:root:i=729 residual=0.0003172366414219141
    DEBUG:root:i=730 residual=0.0003172380675096065
    DEBUG:root:i=731 residual=0.0003172381257172674
    DEBUG:root:i=732 residual=0.0003139354521408677
    DEBUG:root:i=733 residual=0.00031063141068443656
    DEBUG:root:i=734 residual=0.0003139312320854515
    DEBUG:root:i=735 residual=0.0003073285915888846
    DEBUG:root:i=736 residual=0.00030732437153346837
    DEBUG:root:i=737 residual=0.00031062844209372997
    DEBUG:root:i=738 residual=0.00031723384745419025
    DEBUG:root:i=739 residual=0.0003172379801981151
    DEBUG:root:i=740 residual=0.00031723800930194557
    DEBUG:root:i=741 residual=0.00031393542303703725
    DEBUG:root:i=742 residual=0.00031063129426911473
    DEBUG:root:i=743 residual=0.0003139326872769743
    DEBUG:root:i=744 residual=0.0003073272237088531
    DEBUG:root:i=745 residual=0.00030402306583710015
    DEBUG:root:i=746 residual=0.00031723116990178823
    DEBUG:root:i=747 residual=0.00031723800930194557
    DEBUG:root:i=748 residual=0.0003139353939332068
    DEBUG:root:i=749 residual=0.00031723672873340547
    DEBUG:root:i=750 residual=0.0003172379801981151
    DEBUG:root:i=751 residual=0.00031723809661343694
    DEBUG:root:i=752 residual=0.0003172381257172674
    DEBUG:root:i=753 residual=0.00032054082839749753
    DEBUG:root:i=754 residual=0.0003172395227011293
    DEBUG:root:i=755 residual=0.00031723821302875876
    DEBUG:root:i=756 residual=0.0003172380675096065
    DEBUG:root:i=757 residual=0.0003172395518049598
    DEBUG:root:i=758 residual=0.0003172381257172674
    DEBUG:root:i=759 residual=0.00031723815482109785
    DEBUG:root:i=760 residual=0.00031723809661343694
    DEBUG:root:i=761 residual=0.000317238038405776
    DEBUG:root:i=762 residual=0.0003172381257172674
    DEBUG:root:i=763 residual=0.0003139354521408677
    DEBUG:root:i=764 residual=0.00031723672873340547
    DEBUG:root:i=765 residual=0.00031393548124469817
    DEBUG:root:i=766 residual=0.00031063135247677565
    DEBUG:root:i=767 residual=0.0003172352153342217
    DEBUG:root:i=768 residual=0.0003172381257172674
    DEBUG:root:i=769 residual=0.000317238038405776
    DEBUG:root:i=770 residual=0.0003172381257172674
    DEBUG:root:i=771 residual=0.00031723809661343694
    DEBUG:root:i=772 residual=0.00031723809661343694
    DEBUG:root:i=773 residual=0.0003139353939332068
    DEBUG:root:i=774 residual=0.00030732876621186733
    DEBUG:root:i=775 residual=0.0003073244297411293
    DEBUG:root:i=776 residual=0.0003106270742136985
    DEBUG:root:i=777 residual=0.0003139324835501611
    DEBUG:root:i=778 residual=0.0003172365832142532
    DEBUG:root:i=779 residual=0.00031393536482937634
    DEBUG:root:i=780 residual=0.00031723527354188263
    DEBUG:root:i=781 residual=0.00031723949359729886
    DEBUG:root:i=782 residual=0.0003172381257172674
    DEBUG:root:i=783 residual=0.00031723809661343694
    DEBUG:root:i=784 residual=0.0003205407992936671
    DEBUG:root:i=785 residual=0.0003172395227011293
    DEBUG:root:i=786 residual=0.0003172381839249283
    DEBUG:root:i=787 residual=0.00031723809661343694
    DEBUG:root:i=788 residual=0.0003139368782285601
    DEBUG:root:i=789 residual=0.00031723527354188263
    DEBUG:root:i=790 residual=0.00031723809661343694
    DEBUG:root:i=791 residual=0.0003172395227011293
    DEBUG:root:i=792 residual=0.0003172381257172674
    DEBUG:root:i=793 residual=0.0003172381257172674
    DEBUG:root:i=794 residual=0.000317238038405776
    DEBUG:root:i=795 residual=0.0003172381257172674
    DEBUG:root:i=796 residual=0.0003139355103485286
    DEBUG:root:i=797 residual=0.00031723675783723593
    DEBUG:root:i=798 residual=0.00031723809661343694
    DEBUG:root:i=799 residual=0.0003172381257172674
    DEBUG:root:i=800 residual=0.000317238038405776
    DEBUG:root:i=801 residual=0.0003172381257172674
    DEBUG:root:i=802 residual=0.0003172380675096065
    DEBUG:root:i=803 residual=0.0003172381839249283
    DEBUG:root:i=804 residual=0.0003172395227011293
    DEBUG:root:i=805 residual=0.00031723809661343694
    DEBUG:root:i=806 residual=0.00031723815482109785
    DEBUG:root:i=807 residual=0.00031723809661343694
    DEBUG:root:i=808 residual=0.0003172381839249283
    DEBUG:root:i=809 residual=0.0003172380675096065
    DEBUG:root:i=810 residual=0.00031723815482109785
    DEBUG:root:i=811 residual=0.000317238038405776
    DEBUG:root:i=812 residual=0.0003172380675096065
    DEBUG:root:i=813 residual=0.000320542196277529
    DEBUG:root:i=814 residual=0.00031723958090879023
    DEBUG:root:i=815 residual=0.00031723821302875876
    DEBUG:root:i=816 residual=0.000317238038405776
    DEBUG:root:i=817 residual=0.00031723815482109785
    DEBUG:root:i=818 residual=0.00031723809661343694
    DEBUG:root:i=819 residual=0.0003172381257172674
    DEBUG:root:i=820 residual=0.00031723949359729886
    DEBUG:root:i=821 residual=0.0003172381839249283
    DEBUG:root:i=822 residual=0.0003172381257172674
    DEBUG:root:i=823 residual=0.0003172381257172674
    DEBUG:root:i=824 residual=0.0003172381257172674
    DEBUG:root:i=825 residual=0.0003139354521408677
    DEBUG:root:i=826 residual=0.00031723675783723593
    DEBUG:root:i=827 residual=0.000317238038405776
    DEBUG:root:i=828 residual=0.00031723809661343694
    DEBUG:root:i=829 residual=0.00031723809661343694
    DEBUG:root:i=830 residual=0.0003172381839249283
    DEBUG:root:i=831 residual=0.0003172381257172674
    DEBUG:root:i=832 residual=0.00031393684912472963
    DEBUG:root:i=833 residual=0.00030732728191651404
    DEBUG:root:i=834 residual=0.00031062847119756043
    DEBUG:root:i=835 residual=0.00031393117387779057
    DEBUG:root:i=836 residual=0.0003139339096378535
    DEBUG:root:i=837 residual=0.0003139339096378535
    DEBUG:root:i=838 residual=0.00031723533174954355
    DEBUG:root:i=839 residual=0.0003139367327094078
    DEBUG:root:i=840 residual=0.0003106299845967442
    DEBUG:root:i=841 residual=0.00031723527354188263
    DEBUG:root:i=842 residual=0.0003172380675096065
    DEBUG:root:i=843 residual=0.0003172381257172674
    DEBUG:root:i=844 residual=0.00031723809661343694
    DEBUG:root:i=845 residual=0.0003172380675096065
    DEBUG:root:i=846 residual=0.00031723815482109785
    DEBUG:root:i=847 residual=0.0003172381257172674
    DEBUG:root:i=848 residual=0.0003172395518049598
    DEBUG:root:i=849 residual=0.0003172380675096065
    DEBUG:root:i=850 residual=0.0003172381257172674
    DEBUG:root:i=851 residual=0.0003172381257172674
    DEBUG:root:i=852 residual=0.00031723809661343694
    DEBUG:root:i=853 residual=0.0003172381257172674
    DEBUG:root:i=854 residual=0.00031723821302875876
    DEBUG:root:i=855 residual=0.000317238038405776
    DEBUG:root:i=856 residual=0.0003172381257172674
    DEBUG:root:i=857 residual=0.0003172381839249283
    DEBUG:root:i=858 residual=0.0003172394644934684
    DEBUG:root:i=859 residual=0.0003172381839249283
    DEBUG:root:i=860 residual=0.000317238038405776
    DEBUG:root:i=861 residual=0.0003172381257172674
    DEBUG:root:i=862 residual=0.0003172381257172674
    DEBUG:root:i=863 residual=0.0003172381257172674
    DEBUG:root:i=864 residual=0.0003139355103485286
    DEBUG:root:i=865 residual=0.00030732862069271505
    DEBUG:root:i=866 residual=0.0003073244297411293
    DEBUG:root:i=867 residual=0.0003139310865662992
    DEBUG:root:i=868 residual=0.00031393254175782204
    DEBUG:root:i=869 residual=0.00031723667052574456
    DEBUG:root:i=870 residual=0.000317238038405776
    DEBUG:root:i=871 residual=0.0003139353939332068
    DEBUG:root:i=872 residual=0.00031063135247677565
    DEBUG:root:i=873 residual=0.00032053806353360415
    DEBUG:root:i=874 residual=0.0003172394062858075
    DEBUG:root:i=875 residual=0.0003172381839249283
    DEBUG:root:i=876 residual=0.00031723809661343694
    DEBUG:root:i=877 residual=0.0003172381257172674
    DEBUG:root:i=878 residual=0.0003172381257172674
    DEBUG:root:i=879 residual=0.0003172381257172674
    DEBUG:root:i=880 residual=0.00031393684912472963
    DEBUG:root:i=881 residual=0.0003172354190610349
    DEBUG:root:i=882 residual=0.000317238038405776
    DEBUG:root:i=883 residual=0.0003139353939332068
    DEBUG:root:i=884 residual=0.00031063141068443656
    DEBUG:root:i=885 residual=0.0003139325126539916
    DEBUG:root:i=886 residual=0.00030732867890037596
    DEBUG:root:i=887 residual=0.0003106271324213594
    DEBUG:root:i=888 residual=0.0003172351571265608
    DEBUG:root:i=889 residual=0.0003172365832142532
    DEBUG:root:i=890 residual=0.000317238038405776
    DEBUG:root:i=891 residual=0.0003172395227011293
    DEBUG:root:i=892 residual=0.0003172381839249283
    DEBUG:root:i=893 residual=0.000317238038405776
    DEBUG:root:i=894 residual=0.000317238038405776
    DEBUG:root:i=895 residual=0.0003139355103485286
    DEBUG:root:i=896 residual=0.0003106313233729452
    DEBUG:root:i=897 residual=0.0003139312320854515
    DEBUG:root:i=898 residual=0.00030732856248505414
    DEBUG:root:i=899 residual=0.0003073258267249912
    DEBUG:root:i=900 residual=0.00031062704510986805
    DEBUG:root:i=901 residual=0.00031723384745419025
    DEBUG:root:i=902 residual=0.0003139353357255459
    DEBUG:root:i=903 residual=0.0003106313233729452
    DEBUG:root:i=904 residual=0.0003139325708616525
    DEBUG:root:i=905 residual=0.00031062986818142235
    DEBUG:root:i=906 residual=0.00031393259996548295
    DEBUG:root:i=907 residual=0.0003106312651652843
    DEBUG:root:i=908 residual=0.00031393117387779057
    DEBUG:root:i=909 residual=0.00031393388053402305
    DEBUG:root:i=910 residual=0.00031723661231808364
    DEBUG:root:i=911 residual=0.0003139354521408677
    DEBUG:root:i=912 residual=0.00031723672873340547
    DEBUG:root:i=913 residual=0.00031723809661343694
    DEBUG:root:i=914 residual=0.00031723800930194557
    DEBUG:root:i=915 residual=0.00031393548124469817
    DEBUG:root:i=916 residual=0.0003106313233729452
    DEBUG:root:i=917 residual=0.0003139312320854515
    DEBUG:root:i=918 residual=0.00031063129426911473
    DEBUG:root:i=919 residual=0.0003139326290693134
    DEBUG:root:i=920 residual=0.0003073271072935313
    DEBUG:root:i=921 residual=0.00031062852940522134
    DEBUG:root:i=922 residual=0.00031723384745419025
    DEBUG:root:i=923 residual=0.0003172379801981151
    DEBUG:root:i=924 residual=0.00031723815482109785
    DEBUG:root:i=925 residual=0.00031723800930194557
    DEBUG:root:i=926 residual=0.0003172381257172674
    DEBUG:root:i=927 residual=0.0003139353939332068
    DEBUG:root:i=928 residual=0.000310631439788267
    DEBUG:root:i=929 residual=0.0003139325708616525
    DEBUG:root:i=930 residual=0.00030732862069271505
    DEBUG:root:i=931 residual=0.0003073244006372988
    DEBUG:root:i=932 residual=0.00031062704510986805
    DEBUG:root:i=933 residual=0.0003139324835501611
    DEBUG:root:i=934 residual=0.0003106298390775919
    DEBUG:root:i=935 residual=0.0003205379762221128
    DEBUG:root:i=936 residual=0.0003172394062858075
    DEBUG:root:i=937 residual=0.00032054082839749753
    DEBUG:root:i=938 residual=0.0003172409487888217
    DEBUG:root:i=939 residual=0.00031723821302875876
    DEBUG:root:i=940 residual=0.00031723809661343694
    DEBUG:root:i=941 residual=0.00031723809661343694
    DEBUG:root:i=942 residual=0.0003172380675096065
    DEBUG:root:i=943 residual=0.00031723815482109785
    DEBUG:root:i=944 residual=0.0003172381839249283
    DEBUG:root:i=945 residual=0.00031723809661343694
    DEBUG:root:i=946 residual=0.0003139353939332068
    DEBUG:root:i=947 residual=0.00031723667052574456
    DEBUG:root:i=948 residual=0.0003172381257172674
    DEBUG:root:i=949 residual=0.0003172380675096065
    DEBUG:root:i=950 residual=0.00031723958090879023
    DEBUG:root:i=951 residual=0.00032054082839749753
    DEBUG:root:i=952 residual=0.0003172395227011293
    DEBUG:root:i=953 residual=0.0003172381257172674
    DEBUG:root:i=954 residual=0.0003172381257172674
    DEBUG:root:i=955 residual=0.00031393548124469817
    DEBUG:root:i=956 residual=0.00031723672873340547
    DEBUG:root:i=957 residual=0.0003172380675096065
    DEBUG:root:i=958 residual=0.0003172380675096065
    DEBUG:root:i=959 residual=0.00031723958090879023
    DEBUG:root:i=960 residual=0.0003172381257172674
    DEBUG:root:i=961 residual=0.0003172381257172674
    DEBUG:root:i=962 residual=0.00031723809661343694
    DEBUG:root:i=963 residual=0.0003139353939332068
    DEBUG:root:i=964 residual=0.00031723672873340547
    DEBUG:root:i=965 residual=0.00031723809661343694
    DEBUG:root:i=966 residual=0.0003172381257172674
    DEBUG:root:i=967 residual=0.0003172381839249283
    DEBUG:root:i=968 residual=0.00032054074108600616
    DEBUG:root:i=969 residual=0.0003205421380698681
    DEBUG:root:i=970 residual=0.0003172410069964826
    DEBUG:root:i=971 residual=0.0003172381257172674
    DEBUG:root:i=972 residual=0.00031723821302875876
    DEBUG:root:i=973 residual=0.0003172381257172674
    DEBUG:root:i=974 residual=0.0003139354521408677
    DEBUG:root:i=975 residual=0.00031723672873340547
    DEBUG:root:i=976 residual=0.000317238038405776
    DEBUG:root:i=977 residual=0.00031723815482109785
    DEBUG:root:i=978 residual=0.00031723809661343694
    DEBUG:root:i=979 residual=0.00031723958090879023
    DEBUG:root:i=980 residual=0.000317238038405776
    DEBUG:root:i=981 residual=0.0003172381257172674
    DEBUG:root:i=982 residual=0.0003172381257172674
    DEBUG:root:i=983 residual=0.00031723815482109785
    DEBUG:root:i=984 residual=0.000320540857501328
    DEBUG:root:i=985 residual=0.0003172394062858075
    DEBUG:root:i=986 residual=0.00031393690733239055
    DEBUG:root:i=987 residual=0.00031723675783723593
    DEBUG:root:i=988 residual=0.00031723809661343694
    DEBUG:root:i=989 residual=0.00031723809661343694
    DEBUG:root:i=990 residual=0.0003172381257172674
    DEBUG:root:i=991 residual=0.0003172381257172674
    DEBUG:root:i=992 residual=0.0003172381257172674
    DEBUG:root:i=993 residual=0.0003172381257172674
    DEBUG:root:i=994 residual=0.00031723815482109785
    DEBUG:root:i=995 residual=0.0003172381257172674
    DEBUG:root:i=996 residual=0.00031723800930194557
    DEBUG:root:i=997 residual=0.00031723841675557196
    DEBUG:root:i=998 residual=0.0003139368782285601
    DEBUG:root:i=999 residual=0.0003172354190610349
    INFO:root:rank=0 pagerank=1.0624e+01 url=www.lawfareblog.com/snowden-revelations
    INFO:root:rank=1 pagerank=1.0624e+01 url=www.lawfareblog.com/lawfare-job-board
    INFO:root:rank=2 pagerank=1.0624e+01 url=www.lawfareblog.com/masthead
    INFO:root:rank=3 pagerank=1.0624e+01 url=www.lawfareblog.com/litigation-documents-resources-related-travel-ban
    INFO:root:rank=4 pagerank=1.0624e+01 url=www.lawfareblog.com/subscribe-lawfare
    INFO:root:rank=5 pagerank=1.0624e+01 url=www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
    INFO:root:rank=6 pagerank=1.0624e+01 url=www.lawfareblog.com/documents-related-mueller-investigation
    INFO:root:rank=7 pagerank=1.0624e+01 url=www.lawfareblog.com/our-comments-policy
    INFO:root:rank=8 pagerank=1.0624e+01 url=www.lawfareblog.com/upcoming-events
    INFO:root:rank=9 pagerank=1.0624e+01 url=www.lawfareblog.com/topics
   
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2
    DEBUG:root:computing indices
    DEBUG:root:computing values
    DEBUG:root:i=0 residual=5.116616725921631
    DEBUG:root:i=1 residual=2.984739065170288
    DEBUG:root:i=2 residual=2.2717621326446533
    DEBUG:root:i=3 residual=1.5339359045028687
    DEBUG:root:i=4 residual=0.9850409626960754
    DEBUG:root:i=5 residual=0.6557350158691406
    DEBUG:root:i=6 residual=0.4652453064918518
    DEBUG:root:i=7 residual=0.34404414892196655
    DEBUG:root:i=8 residual=0.2585417926311493
    DEBUG:root:i=9 residual=0.195647194981575
    DEBUG:root:i=10 residual=0.14962732791900635
    DEBUG:root:i=11 residual=0.11680074781179428
    DEBUG:root:i=12 residual=0.09400086104869843
    DEBUG:root:i=13 residual=0.0783218964934349
    DEBUG:root:i=14 residual=0.06728129833936691
    DEBUG:root:i=15 residual=0.05901386961340904
    DEBUG:root:i=16 residual=0.052311111241579056
    DEBUG:root:i=17 residual=0.04649629816412926
    DEBUG:root:i=18 residual=0.041234955191612244
    DEBUG:root:i=19 residual=0.036389391869306564
    DEBUG:root:i=20 residual=0.03191784396767616
    DEBUG:root:i=21 residual=0.027816113084554672
    DEBUG:root:i=22 residual=0.024088440462946892
    DEBUG:root:i=23 residual=0.020737752318382263
    DEBUG:root:i=24 residual=0.017756015062332153
    DEBUG:root:i=25 residual=0.01512843556702137
    DEBUG:root:i=26 residual=0.012832536362111568
    DEBUG:root:i=27 residual=0.01084243692457676
    DEBUG:root:i=28 residual=0.009129120036959648
    DEBUG:root:i=29 residual=0.007662927731871605
    DEBUG:root:i=30 residual=0.006415168289095163
    DEBUG:root:i=31 residual=0.005357979331165552
    DEBUG:root:i=32 residual=0.00446558790281415
    DEBUG:root:i=33 residual=0.003715618746355176
    DEBUG:root:i=34 residual=0.003087027929723263
    DEBUG:root:i=35 residual=0.002561302622780204
    DEBUG:root:i=36 residual=0.002122886013239622
    DEBUG:root:i=37 residual=0.0017581103602424264
    DEBUG:root:i=38 residual=0.001454571494832635
    DEBUG:root:i=39 residual=0.001203120336867869
    DEBUG:root:i=40 residual=0.0009945336496457458
    DEBUG:root:i=41 residual=0.0008218936272896826
    DEBUG:root:i=42 residual=0.0006789034232497215
    DEBUG:root:i=43 residual=0.0005607760394923389
    DEBUG:root:i=44 residual=0.00046330533223226666
    DEBUG:root:i=45 residual=0.0003828405460808426
    DEBUG:root:i=46 residual=0.00031630677403882146
    DEBUG:root:i=47 residual=0.00026124384021386504
    DEBUG:root:i=48 residual=0.00021592095436062664
    DEBUG:root:i=49 residual=0.00017844472313299775
    DEBUG:root:i=50 residual=0.00014751612616237253
    DEBUG:root:i=51 residual=0.00012199909542687237
    DEBUG:root:i=52 residual=0.00010094596655108035
    DEBUG:root:i=53 residual=8.342426735907793e-05
    DEBUG:root:i=54 residual=6.913088873261586e-05
    DEBUG:root:i=55 residual=5.722401328966953e-05
    DEBUG:root:i=56 residual=4.7504629037575796e-05
    DEBUG:root:i=57 residual=3.9216098230099306e-05
    DEBUG:root:i=58 residual=3.261367237428203e-05
    DEBUG:root:i=59 residual=2.700979712244589e-05
    DEBUG:root:i=60 residual=2.238958222733345e-05
    DEBUG:root:i=61 residual=1.8524195183999836e-05
    DEBUG:root:i=62 residual=1.536699892312754e-05
    DEBUG:root:i=63 residual=1.2941852219228167e-05
    DEBUG:root:i=64 residual=1.0631928489601705e-05
    DEBUG:root:i=65 residual=8.92855223355582e-06
    DEBUG:root:i=66 residual=7.350837677222444e-06
    DEBUG:root:i=67 residual=6.182938705023844e-06
    DEBUG:root:i=68 residual=5.121226422488689e-06
    DEBUG:root:i=69 residual=4.19895786762936e-06
    DEBUG:root:i=70 residual=3.65417236025678e-06
    DEBUG:root:i=71 residual=2.9140062451915583e-06
    DEBUG:root:i=72 residual=2.359738800805644e-06
    DEBUG:root:i=73 residual=2.3076083834894234e-06
    DEBUG:root:i=74 residual=1.743710981827462e-06
    DEBUG:root:i=75 residual=1.4988713701313827e-06
    DEBUG:root:i=76 residual=1.150764092017198e-06
    DEBUG:root:i=77 residual=1.0613740641929326e-06
    DEBUG:root:i=78 residual=8.774190405347326e-07
    INFO:root:rank=0 pagerank=4.2777e+00 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
    INFO:root:rank=1 pagerank=2.7719e+00 url=www.lawfareblog.com/livestream-nov-21-impeachment-hearings-0
    INFO:root:rank=2 pagerank=2.7535e+00 url=www.lawfareblog.com/opening-statement-david-holmes
    INFO:root:rank=3 pagerank=1.8722e+00 url=www.lawfareblog.com/senate-examines-threats-homeland
    INFO:root:rank=4 pagerank=1.7419e+00 url=www.lawfareblog.com/what-make-first-day-impeachment-hearings
    INFO:root:rank=5 pagerank=1.7412e+00 url=www.lawfareblog.com/livestream-house-armed-services-committee-hearing-f-35-program
    INFO:root:rank=6 pagerank=1.7349e+00 url=www.lawfareblog.com/whats-house-resolution-impeachment
    INFO:root:rank=7 pagerank=1.6385e+00 url=www.lawfareblog.com/congress-us-policy-toward-syria-and-turkey-overview-recent-hearings
    INFO:root:rank=8 pagerank=1.5598e+00 url=www.lawfareblog.com/summary-david-holmess-deposition-testimony
    INFO:root:rank=9 pagerank=9.1273e-01 url=www.lawfareblog.com/events

   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2 --alpha=0.99999
    DEBUG:root:computing indices
    DEBUG:root:computing values
    DEBUG:root:i=0 residual=6.019479751586914
    DEBUG:root:i=1 residual=4.131033897399902
    DEBUG:root:i=2 residual=3.699080228805542
    DEBUG:root:i=3 residual=2.9384074211120605
    DEBUG:root:i=4 residual=2.219911575317383
    DEBUG:root:i=5 residual=1.7385401725769043
    DEBUG:root:i=6 residual=1.451156735420227
    DEBUG:root:i=7 residual=1.2624785900115967
    DEBUG:root:i=8 residual=1.1161377429962158
    DEBUG:root:i=9 residual=0.9936513304710388
    DEBUG:root:i=10 residual=0.8940240144729614
    DEBUG:root:i=11 residual=0.8210335373878479
    DEBUG:root:i=12 residual=0.7773657441139221
    DEBUG:root:i=13 residual=0.7620036005973816
    DEBUG:root:i=14 residual=0.7700909972190857
    DEBUG:root:i=15 residual=0.7946513891220093
    DEBUG:root:i=16 residual=0.8287094831466675
    DEBUG:root:i=17 residual=0.8665599822998047
    DEBUG:root:i=18 residual=0.9041051864624023
    DEBUG:root:i=19 residual=0.9386507868766785
    DEBUG:root:i=20 residual=0.9685821533203125
    DEBUG:root:i=21 residual=0.9930528402328491
    DEBUG:root:i=22 residual=1.0117459297180176
    DEBUG:root:i=23 residual=1.0246989727020264
    DEBUG:root:i=24 residual=1.0321825742721558
    DEBUG:root:i=25 residual=1.0346076488494873
    DEBUG:root:i=26 residual=1.0324623584747314
    DEBUG:root:i=27 residual=1.0262778997421265
    DEBUG:root:i=28 residual=1.0165843963623047
    DEBUG:root:i=29 residual=1.0038970708847046
    DEBUG:root:i=30 residual=0.9887090921401978
    DEBUG:root:i=31 residual=0.9714718461036682
    DEBUG:root:i=32 residual=0.9525952935218811
    DEBUG:root:i=33 residual=0.9324474930763245
    DEBUG:root:i=34 residual=0.9113600850105286
    DEBUG:root:i=35 residual=0.8896127343177795
    DEBUG:root:i=36 residual=0.86745685338974
    DEBUG:root:i=37 residual=0.8451002240180969
    DEBUG:root:i=38 residual=0.8227171897888184
    DEBUG:root:i=39 residual=0.8004683256149292
    DEBUG:root:i=40 residual=0.7784712910652161
    DEBUG:root:i=41 residual=0.756823718547821
    DEBUG:root:i=42 residual=0.7356073260307312
    DEBUG:root:i=43 residual=0.7148909568786621
    DEBUG:root:i=44 residual=0.6947129964828491
    DEBUG:root:i=45 residual=0.6751192212104797
    DEBUG:root:i=46 residual=0.6561205387115479
    DEBUG:root:i=47 residual=0.6377402544021606
    DEBUG:root:i=48 residual=0.6199856996536255
    DEBUG:root:i=49 residual=0.6028539538383484
    DEBUG:root:i=50 residual=0.5863428115844727
    DEBUG:root:i=51 residual=0.5704388618469238
    DEBUG:root:i=52 residual=0.5551407337188721
    DEBUG:root:i=53 residual=0.5404224991798401
    DEBUG:root:i=54 residual=0.5262718200683594
    DEBUG:root:i=55 residual=0.5126755833625793
    DEBUG:root:i=56 residual=0.4996073842048645
    DEBUG:root:i=57 residual=0.4870529770851135
    DEBUG:root:i=58 residual=0.47499164938926697
    DEBUG:root:i=59 residual=0.46339893341064453
    DEBUG:root:i=60 residual=0.45225951075553894
    DEBUG:root:i=61 residual=0.4415546655654907
    DEBUG:root:i=62 residual=0.4312572777271271
    DEBUG:root:i=63 residual=0.42136073112487793
    DEBUG:root:i=64 residual=0.4118330478668213
    DEBUG:root:i=65 residual=0.40267038345336914
    DEBUG:root:i=66 residual=0.39384645223617554
    DEBUG:root:i=67 residual=0.3853493332862854
    DEBUG:root:i=68 residual=0.37716084718704224
    DEBUG:root:i=69 residual=0.369269460439682
    DEBUG:root:i=70 residual=0.36165687441825867
    DEBUG:root:i=71 residual=0.3543132245540619
    DEBUG:root:i=72 residual=0.3472256660461426
    DEBUG:root:i=73 residual=0.34037894010543823
    DEBUG:root:i=74 residual=0.33376166224479675
    DEBUG:root:i=75 residual=0.3273681104183197
    DEBUG:root:i=76 residual=0.32118165493011475
    DEBUG:root:i=77 residual=0.315193772315979
    DEBUG:root:i=78 residual=0.3094014823436737
    DEBUG:root:i=79 residual=0.30378812551498413
    DEBUG:root:i=80 residual=0.2983468472957611
    DEBUG:root:i=81 residual=0.29307857155799866
    DEBUG:root:i=82 residual=0.28796300292015076
    DEBUG:root:i=83 residual=0.28300097584724426
    DEBUG:root:i=84 residual=0.27818042039871216
    DEBUG:root:i=85 residual=0.2735048234462738
    DEBUG:root:i=86 residual=0.2689540684223175
    DEBUG:root:i=87 residual=0.264534592628479
    DEBUG:root:i=88 residual=0.2602340877056122
    DEBUG:root:i=89 residual=0.2560523748397827
    DEBUG:root:i=90 residual=0.2519865930080414
    DEBUG:root:i=91 residual=0.24801789224147797
    DEBUG:root:i=92 residual=0.2441646307706833
    DEBUG:root:i=93 residual=0.24039486050605774
    DEBUG:root:i=94 residual=0.23672692477703094
    DEBUG:root:i=95 residual=0.23315788805484772
    DEBUG:root:i=96 residual=0.22966401278972626
    DEBUG:root:i=97 residual=0.22626350820064545
    DEBUG:root:i=98 residual=0.22293764352798462
    DEBUG:root:i=99 residual=0.2196970134973526
    DEBUG:root:i=100 residual=0.21652549505233765
    DEBUG:root:i=101 residual=0.21342821419239044
    DEBUG:root:i=102 residual=0.2104104608297348
    DEBUG:root:i=103 residual=0.20745094120502472
    DEBUG:root:i=104 residual=0.20455747842788696
    DEBUG:root:i=105 residual=0.2017219513654709
    DEBUG:root:i=106 residual=0.1989602893590927
    DEBUG:root:i=107 residual=0.19624574482440948
    DEBUG:root:i=108 residual=0.1935889720916748
    DEBUG:root:i=109 residual=0.19098982214927673
    DEBUG:root:i=110 residual=0.1884455680847168
    DEBUG:root:i=111 residual=0.1859535127878189
    DEBUG:root:i=112 residual=0.18350829184055328
    DEBUG:root:i=113 residual=0.18111513555049896
    DEBUG:root:i=114 residual=0.17876607179641724
    DEBUG:root:i=115 residual=0.17645834386348724
    DEBUG:root:i=116 residual=0.17421025037765503
    DEBUG:root:i=117 residual=0.17198504507541656
    DEBUG:root:i=118 residual=0.16981427371501923
    DEBUG:root:i=119 residual=0.16767661273479462
    DEBUG:root:i=120 residual=0.16558805108070374
    DEBUG:root:i=121 residual=0.16352730989456177
    DEBUG:root:i=122 residual=0.1615048497915268
    DEBUG:root:i=123 residual=0.15952077507972717
    DEBUG:root:i=124 residual=0.15756438672542572
    DEBUG:root:i=125 residual=0.15564623475074768
    DEBUG:root:i=126 residual=0.15377405285835266
    DEBUG:root:i=127 residual=0.15191905200481415
    DEBUG:root:i=128 residual=0.15009693801403046
    DEBUG:root:i=129 residual=0.14831013977527618
    DEBUG:root:i=130 residual=0.1465483009815216
    DEBUG:root:i=131 residual=0.14481382071971893
    DEBUG:root:i=132 residual=0.1431121528148651
    DEBUG:root:i=133 residual=0.14144055545330048
    DEBUG:root:i=134 residual=0.13979361951351166
    DEBUG:root:i=135 residual=0.13816361129283905
    DEBUG:root:i=136 residual=0.13656607270240784
    DEBUG:root:i=137 residual=0.13499854505062103
    DEBUG:root:i=138 residual=0.13345302641391754
    DEBUG:root:i=139 residual=0.1319267749786377
    DEBUG:root:i=140 residual=0.13042527437210083
    DEBUG:root:i=141 residual=0.1289403885602951
    DEBUG:root:i=142 residual=0.12748797237873077
    DEBUG:root:i=143 residual=0.1260574609041214
    DEBUG:root:i=144 residual=0.12463831156492233
    DEBUG:root:i=145 residual=0.12324109673500061
    DEBUG:root:i=146 residual=0.1218709796667099
    DEBUG:root:i=147 residual=0.120522640645504
    DEBUG:root:i=148 residual=0.11918042600154877
    DEBUG:root:i=149 residual=0.1178705170750618
    DEBUG:root:i=150 residual=0.11656925827264786
    DEBUG:root:i=151 residual=0.11528978496789932
    DEBUG:root:i=152 residual=0.11402948945760727
    DEBUG:root:i=153 residual=0.11277776211500168
    DEBUG:root:i=154 residual=0.11155566573143005
    DEBUG:root:i=155 residual=0.11034214496612549
    DEBUG:root:i=156 residual=0.10915026813745499
    DEBUG:root:i=157 residual=0.10797486454248428
    DEBUG:root:i=158 residual=0.10680284351110458
    DEBUG:root:i=159 residual=0.10566293448209763
    DEBUG:root:i=160 residual=0.10453411936759949
    DEBUG:root:i=161 residual=0.10341653227806091
    DEBUG:root:i=162 residual=0.10232056677341461
    DEBUG:root:i=163 residual=0.10121994465589523
    DEBUG:root:i=164 residual=0.10014882683753967
    DEBUG:root:i=165 residual=0.0990862026810646
    DEBUG:root:i=166 residual=0.09804248064756393
    DEBUG:root:i=167 residual=0.09700731188058853
    DEBUG:root:i=168 residual=0.09599370509386063
    DEBUG:root:i=169 residual=0.09497545659542084
    DEBUG:root:i=170 residual=0.09398388862609863
    DEBUG:root:i=171 residual=0.09299559146165848
    DEBUG:root:i=172 residual=0.09202361851930618
    DEBUG:root:i=173 residual=0.09106528759002686
    DEBUG:root:i=174 residual=0.09011790156364441
    DEBUG:root:i=175 residual=0.08917377889156342
    DEBUG:root:i=176 residual=0.08825110644102097
    DEBUG:root:i=177 residual=0.08734209835529327
    DEBUG:root:i=178 residual=0.08643616735935211
    DEBUG:root:i=179 residual=0.08554389327764511
    DEBUG:root:i=180 residual=0.08466263115406036
    DEBUG:root:i=181 residual=0.08378449082374573
    DEBUG:root:i=182 residual=0.0829302966594696
    DEBUG:root:i=183 residual=0.08207933604717255
    DEBUG:root:i=184 residual=0.08123407512903214
    DEBUG:root:i=185 residual=0.08040235936641693
    DEBUG:root:i=186 residual=0.07958155870437622
    DEBUG:root:i=187 residual=0.07877437770366669
    DEBUG:root:i=188 residual=0.07795984297990799
    DEBUG:root:i=189 residual=0.07716663926839828
    DEBUG:root:i=190 residual=0.07638436555862427
    DEBUG:root:i=191 residual=0.07561302930116653
    DEBUG:root:i=192 residual=0.07483691722154617
    DEBUG:root:i=193 residual=0.07409001886844635
    DEBUG:root:i=194 residual=0.07333050668239594
    DEBUG:root:i=195 residual=0.0725870206952095
    DEBUG:root:i=196 residual=0.07185978442430496
    DEBUG:root:i=197 residual=0.07112764567136765
    DEBUG:root:i=198 residual=0.07041170448064804
    DEBUG:root:i=199 residual=0.0697038546204567
    DEBUG:root:i=200 residual=0.06900965422391891
    DEBUG:root:i=201 residual=0.06831055879592896
    DEBUG:root:i=202 residual=0.06762243062257767
    DEBUG:root:i=203 residual=0.06694504618644714
    DEBUG:root:i=204 residual=0.06627330183982849
    DEBUG:root:i=205 residual=0.06560979783535004
    DEBUG:root:i=206 residual=0.06495193392038345
    DEBUG:root:i=207 residual=0.06430492550134659
    DEBUG:root:i=208 residual=0.06366346776485443
    DEBUG:root:i=209 residual=0.06303027272224426
    DEBUG:root:i=210 residual=0.06239733472466469
    DEBUG:root:i=211 residual=0.061783164739608765
    DEBUG:root:i=212 residual=0.061156295239925385
    DEBUG:root:i=213 residual=0.06054804101586342
    DEBUG:root:i=214 residual=0.059947989881038666
    DEBUG:root:i=215 residual=0.05935347080230713
    DEBUG:root:i=216 residual=0.05875931307673454
    DEBUG:root:i=217 residual=0.05818391218781471
    DEBUG:root:i=218 residual=0.05760348588228226
    DEBUG:root:i=219 residual=0.057028722018003464
    DEBUG:root:i=220 residual=0.056464698165655136
    DEBUG:root:i=221 residual=0.05589830502867699
    DEBUG:root:i=222 residual=0.05535067245364189
    DEBUG:root:i=223 residual=0.054808489978313446
    DEBUG:root:i=224 residual=0.0542588047683239
    DEBUG:root:i=225 residual=0.05372513085603714
    DEBUG:root:i=226 residual=0.05319967493414879
    DEBUG:root:i=227 residual=0.05266653373837471
    DEBUG:root:i=228 residual=0.052149463444948196
    DEBUG:root:i=229 residual=0.05163796618580818
    DEBUG:root:i=230 residual=0.051126737147569656
    DEBUG:root:i=231 residual=0.05062370002269745
    DEBUG:root:i=232 residual=0.0501287505030632
    DEBUG:root:i=233 residual=0.04963664710521698
    DEBUG:root:i=234 residual=0.049139603972435
    DEBUG:root:i=235 residual=0.04865597188472748
    DEBUG:root:i=236 residual=0.048185721039772034
    DEBUG:root:i=237 residual=0.04770781099796295
    DEBUG:root:i=238 residual=0.04723814129829407
    DEBUG:root:i=239 residual=0.04677649214863777
    DEBUG:root:i=240 residual=0.04631515219807625
    DEBUG:root:i=241 residual=0.04585922509431839
    DEBUG:root:i=242 residual=0.045414067804813385
    DEBUG:root:i=243 residual=0.04496922343969345
    DEBUG:root:i=244 residual=0.04452201724052429
    DEBUG:root:i=245 residual=0.04408549517393112
    DEBUG:root:i=246 residual=0.04365181550383568
    DEBUG:root:i=247 residual=0.04322629049420357
    DEBUG:root:i=248 residual=0.042798321694135666
    DEBUG:root:i=249 residual=0.042386263608932495
    DEBUG:root:i=250 residual=0.0419771783053875
    DEBUG:root:i=251 residual=0.04156303033232689
    DEBUG:root:i=252 residual=0.041156962513923645
    DEBUG:root:i=253 residual=0.040748562663793564
    DEBUG:root:i=254 residual=0.04035607725381851
    DEBUG:root:i=255 residual=0.039961207658052444
    DEBUG:root:i=256 residual=0.039563875645399094
    DEBUG:root:i=257 residual=0.03917989879846573
    DEBUG:root:i=258 residual=0.03879352658987045
    DEBUG:root:i=259 residual=0.03842306137084961
    DEBUG:root:i=260 residual=0.03803975135087967
    DEBUG:root:i=261 residual=0.03767760470509529
    DEBUG:root:i=262 residual=0.0373103953897953
    DEBUG:root:i=263 residual=0.03694607689976692
    DEBUG:root:i=264 residual=0.036587122827768326
    DEBUG:root:i=265 residual=0.03622838109731674
    DEBUG:root:i=266 residual=0.035875074565410614
    DEBUG:root:i=267 residual=0.03552205115556717
    DEBUG:root:i=268 residual=0.0351848378777504
    DEBUG:root:i=269 residual=0.03483742102980614
    DEBUG:root:i=270 residual=0.03450322523713112
    DEBUG:root:i=271 residual=0.03416144847869873
    DEBUG:root:i=272 residual=0.033830273896455765
    DEBUG:root:i=273 residual=0.033501919358968735
    DEBUG:root:i=274 residual=0.033173803240060806
    DEBUG:root:i=275 residual=0.032845765352249146
    DEBUG:root:i=276 residual=0.0325336679816246
    DEBUG:root:i=277 residual=0.032216548919677734
    DEBUG:root:i=278 residual=0.031907450407743454
    DEBUG:root:i=279 residual=0.031588081270456314
    DEBUG:root:i=280 residual=0.03128191456198692
    DEBUG:root:i=281 residual=0.030975988134741783
    DEBUG:root:i=282 residual=0.030683251097798347
    DEBUG:root:i=283 residual=0.030380267649888992
    DEBUG:root:i=284 residual=0.030093135312199593
    DEBUG:root:i=285 residual=0.029795734211802483
    DEBUG:root:i=286 residual=0.029503677040338516
    DEBUG:root:i=287 residual=0.02922232449054718
    DEBUG:root:i=288 residual=0.028938477858901024
    DEBUG:root:i=289 residual=0.028654800727963448
    DEBUG:root:i=290 residual=0.02837391570210457
    DEBUG:root:i=291 residual=0.028098439797759056
    DEBUG:root:i=292 residual=0.027820520102977753
    DEBUG:root:i=293 residual=0.02755315974354744
    DEBUG:root:i=294 residual=0.02729119174182415
    DEBUG:root:i=295 residual=0.027021557092666626
    DEBUG:root:i=296 residual=0.026765113696455956
    DEBUG:root:i=297 residual=0.026503605768084526
    DEBUG:root:i=298 residual=0.026252731680870056
    DEBUG:root:i=299 residual=0.025994181632995605
    DEBUG:root:i=300 residual=0.02574094757437706
    DEBUG:root:i=301 residual=0.02549056150019169
    DEBUG:root:i=302 residual=0.025250740349292755
    DEBUG:root:i=303 residual=0.024997947737574577
    DEBUG:root:i=304 residual=0.02475844882428646
    DEBUG:root:i=305 residual=0.02451643906533718
    DEBUG:root:i=306 residual=0.02428235113620758
    DEBUG:root:i=307 residual=0.024045893922448158
    DEBUG:root:i=308 residual=0.02381213568150997
    DEBUG:root:i=309 residual=0.02357853204011917
    DEBUG:root:i=310 residual=0.023350264877080917
    DEBUG:root:i=311 residual=0.023124678060412407
    DEBUG:root:i=312 residual=0.022899338975548744
    DEBUG:root:i=313 residual=0.022676702588796616
    DEBUG:root:i=314 residual=0.022454196587204933
    DEBUG:root:i=315 residual=0.02224493771791458
    DEBUG:root:i=316 residual=0.022027907893061638
    DEBUG:root:i=317 residual=0.021821491420269012
    DEBUG:root:i=318 residual=0.02160732075572014
    DEBUG:root:i=319 residual=0.021395917981863022
    DEBUG:root:i=320 residual=0.021184666082262993
    DEBUG:root:i=321 residual=0.020983917638659477
    DEBUG:root:i=322 residual=0.020772898569703102
    DEBUG:root:i=323 residual=0.020575018599629402
    DEBUG:root:i=324 residual=0.020379912108182907
    DEBUG:root:i=325 residual=0.020182298496365547
    DEBUG:root:i=326 residual=0.0199821088463068
    DEBUG:root:i=327 residual=0.01979520171880722
    DEBUG:root:i=328 residual=0.019603149965405464
    DEBUG:root:i=329 residual=0.019413834437727928
    DEBUG:root:i=330 residual=0.019227221608161926
    DEBUG:root:i=331 residual=0.019038133323192596
    DEBUG:root:i=332 residual=0.018856961280107498
    DEBUG:root:i=333 residual=0.01867332123219967
    DEBUG:root:i=334 residual=0.018497589975595474
    DEBUG:root:i=335 residual=0.018311548978090286
    DEBUG:root:i=336 residual=0.01813609153032303
    DEBUG:root:i=337 residual=0.017955485731363297
    DEBUG:root:i=338 residual=0.017777560278773308
    DEBUG:root:i=339 residual=0.017610210925340652
    DEBUG:root:i=340 residual=0.01744033396244049
    DEBUG:root:i=341 residual=0.017267977818846703
    DEBUG:root:i=342 residual=0.017103586345911026
    DEBUG:root:i=343 residual=0.016939273104071617
    DEBUG:root:i=344 residual=0.016774950549006462
    DEBUG:root:i=345 residual=0.016618667170405388
    DEBUG:root:i=346 residual=0.016457222402095795
    DEBUG:root:i=347 residual=0.0162959024310112
    DEBUG:root:i=348 residual=0.016142498701810837
    DEBUG:root:i=349 residual=0.015978770330548286
    DEBUG:root:i=350 residual=0.015820331871509552
    DEBUG:root:i=351 residual=0.015672393143177032
    DEBUG:root:i=352 residual=0.015521981753408909
    DEBUG:root:i=353 residual=0.015374304726719856
    DEBUG:root:i=354 residual=0.015224084258079529
    DEBUG:root:i=355 residual=0.0150765310972929
    DEBUG:root:i=356 residual=0.014931650832295418
    DEBUG:root:i=357 residual=0.014792191796004772
    DEBUG:root:i=358 residual=0.014639658853411674
    DEBUG:root:i=359 residual=0.014502914622426033
    DEBUG:root:i=360 residual=0.014366235584020615
    DEBUG:root:i=361 residual=0.014219233766198158
    DEBUG:root:i=362 residual=0.014087960124015808
    DEBUG:root:i=363 residual=0.013948969542980194
    DEBUG:root:i=364 residual=0.01381787471473217
    DEBUG:root:i=365 residual=0.013681609183549881
    DEBUG:root:i=366 residual=0.013553224503993988
    DEBUG:root:i=367 residual=0.013417133130133152
    DEBUG:root:i=368 residual=0.013288930058479309
    DEBUG:root:i=369 residual=0.013160843402147293
    DEBUG:root:i=370 residual=0.013038038276135921
    DEBUG:root:i=371 residual=0.012910118326544762
    DEBUG:root:i=372 residual=0.012790014035999775
    DEBUG:root:i=373 residual=0.012657041661441326
    DEBUG:root:i=374 residual=0.012537119910120964
    DEBUG:root:i=375 residual=0.012409462593495846
    DEBUG:root:i=376 residual=0.012297616340219975
    DEBUG:root:i=377 residual=0.012172679416835308
    DEBUG:root:i=378 residual=0.01205306127667427
    DEBUG:root:i=379 residual=0.011943962424993515
    DEBUG:root:i=380 residual=0.011827140115201473
    DEBUG:root:i=381 residual=0.011715594679117203
    DEBUG:root:i=382 residual=0.011601461097598076
    DEBUG:root:i=383 residual=0.011484848335385323
    DEBUG:root:i=384 residual=0.011378725059330463
    DEBUG:root:i=385 residual=0.011270022951066494
    DEBUG:root:i=386 residual=0.011156228370964527
    DEBUG:root:i=387 residual=0.011045074090361595
    DEBUG:root:i=388 residual=0.010947026312351227
    DEBUG:root:i=389 residual=0.010838652029633522
    DEBUG:root:i=390 residual=0.010730236768722534
    DEBUG:root:i=391 residual=0.010629850439727306
    DEBUG:root:i=392 residual=0.010526810772716999
    DEBUG:root:i=393 residual=0.010426497086882591
    DEBUG:root:i=394 residual=0.010321003384888172
    DEBUG:root:i=395 residual=0.010223321616649628
    DEBUG:root:i=396 residual=0.010125813074409962
    DEBUG:root:i=397 residual=0.010033500380814075
    DEBUG:root:i=398 residual=0.009930910542607307
    DEBUG:root:i=399 residual=0.009841348975896835
    DEBUG:root:i=400 residual=0.00974145159125328
    DEBUG:root:i=401 residual=0.009652042761445045
    DEBUG:root:i=402 residual=0.009557445533573627
    DEBUG:root:i=403 residual=0.009457656182348728
    DEBUG:root:i=404 residual=0.009365818463265896
    DEBUG:root:i=405 residual=0.009279235266149044
    DEBUG:root:i=406 residual=0.009190142154693604
    DEBUG:root:i=407 residual=0.009098413400352001
    DEBUG:root:i=408 residual=0.009014621376991272
    DEBUG:root:i=409 residual=0.00892825797200203
    DEBUG:root:i=410 residual=0.008844524621963501
    DEBUG:root:i=411 residual=0.00875307247042656
    DEBUG:root:i=412 residual=0.008669476956129074
    DEBUG:root:i=413 residual=0.00858070608228445
    DEBUG:root:i=414 residual=0.008499792777001858
    DEBUG:root:i=415 residual=0.008421530947089195
    DEBUG:root:i=416 residual=0.00834075640887022
    DEBUG:root:i=417 residual=0.008252186700701714
    DEBUG:root:i=418 residual=0.008181924931704998
    DEBUG:root:i=419 residual=0.00810393039137125
    DEBUG:root:i=420 residual=0.008025969378650188
    DEBUG:root:i=421 residual=0.007947955280542374
    DEBUG:root:i=422 residual=0.00787274818867445
    DEBUG:root:i=423 residual=0.00779225816950202
    DEBUG:root:i=424 residual=0.007722317706793547
    DEBUG:root:i=425 residual=0.007644515484571457
    DEBUG:root:i=426 residual=0.007569409441202879
    DEBUG:root:i=427 residual=0.00749440910294652
    DEBUG:root:i=428 residual=0.007429822348058224
    DEBUG:root:i=429 residual=0.007349655032157898
    DEBUG:root:i=430 residual=0.007279891520738602
    DEBUG:root:i=431 residual=0.007205006200820208
    DEBUG:root:i=432 residual=0.007138052023947239
    DEBUG:root:i=433 residual=0.007065761834383011
    DEBUG:root:i=434 residual=0.0070040845312178135
    DEBUG:root:i=435 residual=0.006929392460733652
    DEBUG:root:i=436 residual=0.006870381999760866
    DEBUG:root:i=437 residual=0.006800959352403879
    DEBUG:root:i=438 residual=0.006739387288689613
    DEBUG:root:i=439 residual=0.0066753108985722065
    DEBUG:root:i=440 residual=0.006603407207876444
    DEBUG:root:i=441 residual=0.006539373658597469
    DEBUG:root:i=442 residual=0.006483209319412708
    DEBUG:root:i=443 residual=0.006416614633053541
    DEBUG:root:i=444 residual=0.006355338729918003
    DEBUG:root:i=445 residual=0.006294026039540768
    DEBUG:root:i=446 residual=0.0062380204908549786
    DEBUG:root:i=447 residual=0.006171592976897955
    DEBUG:root:i=448 residual=0.006110410671681166
    DEBUG:root:i=449 residual=0.006054554134607315
    DEBUG:root:i=450 residual=0.005993429105728865
    DEBUG:root:i=451 residual=0.0059376186691224575
    DEBUG:root:i=452 residual=0.005879220552742481
    DEBUG:root:i=453 residual=0.00582606578245759
    DEBUG:root:i=454 residual=0.0057677035219967365
    DEBUG:root:i=455 residual=0.005709446966648102
    DEBUG:root:i=456 residual=0.005656341556459665
    DEBUG:root:i=457 residual=0.005600726697593927
    DEBUG:root:i=458 residual=0.005552980583161116
    DEBUG:root:i=459 residual=0.005494800396263599
    DEBUG:root:i=460 residual=0.005441865883767605
    DEBUG:root:i=461 residual=0.005388962104916573
    DEBUG:root:i=462 residual=0.005333524197340012
    DEBUG:root:i=463 residual=0.005283280275762081
    DEBUG:root:i=464 residual=0.005238334182649851
    DEBUG:root:i=465 residual=0.00518034165725112
    DEBUG:root:i=466 residual=0.005135438870638609
    DEBUG:root:i=467 residual=0.005080107599496841
    DEBUG:root:i=468 residual=0.005040549673140049
    DEBUG:root:i=469 residual=0.004990478977560997
    DEBUG:root:i=470 residual=0.004940494894981384
    DEBUG:root:i=471 residual=0.004893156699836254
    DEBUG:root:i=472 residual=0.004840534646064043
    DEBUG:root:i=473 residual=0.0048011289909482
    DEBUG:root:i=474 residual=0.004751215688884258
    DEBUG:root:i=475 residual=0.004706565290689468
    DEBUG:root:i=476 residual=0.004656695295125246
    DEBUG:root:i=477 residual=0.004617362283170223
    DEBUG:root:i=478 residual=0.004562376532703638
    DEBUG:root:i=479 residual=0.004528307821601629
    DEBUG:root:i=480 residual=0.0044837999157607555
    DEBUG:root:i=481 residual=0.004444547463208437
    DEBUG:root:i=482 residual=0.004394856281578541
    DEBUG:root:i=483 residual=0.004360858350992203
    DEBUG:root:i=484 residual=0.004313879180699587
    DEBUG:root:i=485 residual=0.0042772977612912655
    DEBUG:root:i=486 residual=0.004227743949741125
    DEBUG:root:i=487 residual=0.004196503199636936
    DEBUG:root:i=488 residual=0.00414958456531167
    DEBUG:root:i=489 residual=0.004113139118999243
    DEBUG:root:i=490 residual=0.0040662530809640884
    DEBUG:root:i=491 residual=0.0040350803174078465
    DEBUG:root:i=492 residual=0.003988300915807486
    DEBUG:root:i=493 residual=0.0039545344188809395
    DEBUG:root:i=494 residual=0.003915618639439344
    DEBUG:root:i=495 residual=0.0038766523357480764
    DEBUG:root:i=496 residual=0.003837710479274392
    DEBUG:root:i=497 residual=0.0038014575839042664
    DEBUG:root:i=498 residual=0.003765217261388898
    DEBUG:root:i=499 residual=0.0037263818085193634
    DEBUG:root:i=500 residual=0.00369546702131629
    DEBUG:root:i=501 residual=0.003656666725873947
    DEBUG:root:i=502 residual=0.0036283256486058235
    DEBUG:root:i=503 residual=0.0035948404110968113
    DEBUG:root:i=504 residual=0.003550925524905324
    DEBUG:root:i=505 residual=0.003517391160130501
    DEBUG:root:i=506 residual=0.003483958076685667
    DEBUG:root:i=507 residual=0.0034531427081674337
    DEBUG:root:i=508 residual=0.00342494435608387
    DEBUG:root:i=509 residual=0.0033810995519161224
    DEBUG:root:i=510 residual=0.003355615306645632
    DEBUG:root:i=511 residual=0.003317014081403613
    DEBUG:root:i=512 residual=0.003288879059255123
    DEBUG:root:i=513 residual=0.0032608150504529476
    DEBUG:root:i=514 residual=0.0032223276793956757
    DEBUG:root:i=515 residual=0.0031916138250380754
    DEBUG:root:i=516 residual=0.0031662124674767256
    DEBUG:root:i=517 residual=0.003130369121208787
    DEBUG:root:i=518 residual=0.003102367976680398
    DEBUG:root:i=519 residual=0.003074383595958352
    DEBUG:root:i=520 residual=0.003041190793737769
    DEBUG:root:i=521 residual=0.0030132264364510775
    DEBUG:root:i=522 residual=0.002985350787639618
    DEBUG:root:i=523 residual=0.0029574150685220957
    DEBUG:root:i=524 residual=0.002929563168436289
    DEBUG:root:i=525 residual=0.0029016428161412477
    DEBUG:root:i=526 residual=0.0028738072142004967
    DEBUG:root:i=527 residual=0.0028460053727030754
    DEBUG:root:i=528 residual=0.002812899649143219
    DEBUG:root:i=529 residual=0.0027903150767087936
    DEBUG:root:i=530 residual=0.002767756348475814
    DEBUG:root:i=531 residual=0.002739979885518551
    DEBUG:root:i=532 residual=0.0027122104074805975
    DEBUG:root:i=533 residual=0.002679303986951709
    DEBUG:root:i=534 residual=0.0026593925431370735
    DEBUG:root:i=535 residual=0.0026342691853642464
    DEBUG:root:i=536 residual=0.0026040019001811743
    DEBUG:root:i=537 residual=0.002584123285487294
    DEBUG:root:i=538 residual=0.002556483494117856
    DEBUG:root:i=539 residual=0.002531397854909301
    DEBUG:root:i=540 residual=0.0025090111885219812
    DEBUG:root:i=541 residual=0.0024892440997064114
    DEBUG:root:i=542 residual=0.002459029434248805
    DEBUG:root:i=543 residual=0.002431444590911269
    DEBUG:root:i=544 residual=0.0024090870283544064
    DEBUG:root:i=545 residual=0.0023867522832006216
    DEBUG:root:i=546 residual=0.002361798193305731
    DEBUG:root:i=547 residual=0.002342085586860776
    DEBUG:root:i=548 residual=0.002319767139852047
    DEBUG:root:i=549 residual=0.0022922370117157698
    DEBUG:root:i=550 residual=0.002272612415254116
    DEBUG:root:i=551 residual=0.002252939622849226
    DEBUG:root:i=552 residual=0.002233346225693822
    DEBUG:root:i=553 residual=0.0022136876359581947
    DEBUG:root:i=554 residual=0.0021940916776657104
    DEBUG:root:i=555 residual=0.002166690304875374
    DEBUG:root:i=556 residual=0.002147051738575101
    DEBUG:root:i=557 residual=0.002124884631484747
    DEBUG:root:i=558 residual=0.0021053317468613386
    DEBUG:root:i=559 residual=0.0020857984200119972
    DEBUG:root:i=560 residual=0.0020662695169448853
    DEBUG:root:i=561 residual=0.002044130815193057
    DEBUG:root:i=562 residual=0.002024613553658128
    DEBUG:root:i=563 residual=0.002007711213082075
    DEBUG:root:i=564 residual=0.0019830570090562105
    DEBUG:root:i=565 residual=0.0019609536975622177
    DEBUG:root:i=566 residual=0.001944078947417438
    DEBUG:root:i=567 residual=0.0019220620160922408
    DEBUG:root:i=568 residual=0.0019078070763498545
    DEBUG:root:i=569 residual=0.001891030464321375
    DEBUG:root:i=570 residual=0.001866343547590077
    DEBUG:root:i=571 residual=0.0018469778588041663
    DEBUG:root:i=572 residual=0.0018327473662793636
    DEBUG:root:i=573 residual=0.0018107770010828972
    DEBUG:root:i=574 residual=0.0017992621287703514
    DEBUG:root:i=575 residual=0.0017799073830246925
    DEBUG:root:i=576 residual=0.001765718450769782
    DEBUG:root:i=577 residual=0.0017489942256361246
    DEBUG:root:i=578 residual=0.0017296687001362443
    DEBUG:root:i=579 residual=0.0017181870061904192
    DEBUG:root:i=580 residual=0.001696252729743719
    DEBUG:root:i=581 residual=0.0016795520205050707
    DEBUG:root:i=582 residual=0.001662928843870759
    DEBUG:root:i=583 residual=0.0016514676390215755
    DEBUG:root:i=584 residual=0.0016347902128472924
    DEBUG:root:i=585 residual=0.0016181257087737322
    DEBUG:root:i=586 residual=0.0015989025123417377
    DEBUG:root:i=587 residual=0.0015822440618649125
    DEBUG:root:i=588 residual=0.0015734288608655334
    DEBUG:root:i=589 residual=0.001556847244501114
    DEBUG:root:i=590 residual=0.0015402210410684347
    DEBUG:root:i=591 residual=0.0015236521139740944
    DEBUG:root:i=592 residual=0.0015044767642393708
    DEBUG:root:i=593 residual=0.0015009205089882016
    DEBUG:root:i=594 residual=0.0014843635726720095
    DEBUG:root:i=595 residual=0.001470428891479969
    DEBUG:root:i=596 residual=0.0014433725737035275
    DEBUG:root:i=597 residual=0.0014425176195800304
    DEBUG:root:i=598 residual=0.0014259907184168696
    DEBUG:root:i=599 residual=0.0014094709185883403
    DEBUG:root:i=600 residual=0.0013929527485743165
    DEBUG:root:i=601 residual=0.0013868948444724083
    DEBUG:root:i=602 residual=0.0013703897129744291
    DEBUG:root:i=603 residual=0.0013538920320570469
    DEBUG:root:i=604 residual=0.0013426259392872453
    DEBUG:root:i=605 residual=0.001331368344835937
    DEBUG:root:i=606 residual=0.0013148850994184613
    DEBUG:root:i=607 residual=0.001308855484239757
    DEBUG:root:i=608 residual=0.0012872255174443126
    DEBUG:root:i=609 residual=0.0012759835226461291
    DEBUG:root:i=610 residual=0.0012699792860075831
    DEBUG:root:i=611 residual=0.0012535819550976157
    DEBUG:root:i=612 residual=0.0012423543957993388
    DEBUG:root:i=613 residual=0.0012311278842389584
    DEBUG:root:i=614 residual=0.0012199790216982365
    DEBUG:root:i=615 residual=0.0012035478139296174
    DEBUG:root:i=616 residual=0.001197623903863132
    DEBUG:root:i=617 residual=0.0011864285916090012
    DEBUG:root:i=618 residual=0.0011752927675843239
    DEBUG:root:i=619 residual=0.0011640953598544002
    DEBUG:root:i=620 residual=0.001147748902440071
    DEBUG:root:i=621 residual=0.0011470807949081063
    DEBUG:root:i=622 residual=0.0011254546698182821
    DEBUG:root:i=623 residual=0.0011247945949435234
    DEBUG:root:i=624 residual=0.001103243208490312
    DEBUG:root:i=625 residual=0.0010973572498187423
    DEBUG:root:i=626 residual=0.0010862700873985887
    DEBUG:root:i=627 residual=0.001080336864106357
    DEBUG:root:i=628 residual=0.0010640218388289213
    DEBUG:root:i=629 residual=0.00105294247623533
    DEBUG:root:i=630 residual=0.0010444867657497525
    DEBUG:root:i=631 residual=0.0010333969257771969
    DEBUG:root:i=632 residual=0.0010275591630488634
    DEBUG:root:i=633 residual=0.0010060505010187626
    DEBUG:root:i=634 residual=0.0010028134565800428
    DEBUG:root:i=635 residual=0.0009943721815943718
    DEBUG:root:i=636 residual=0.0009859908604994416
    DEBUG:root:i=637 residual=0.0009749418823048472
    DEBUG:root:i=638 residual=0.0009717429638840258
    DEBUG:root:i=639 residual=0.000958086340688169
    DEBUG:root:i=640 residual=0.0009496544953435659
    DEBUG:root:i=641 residual=0.0009360790136270225
    DEBUG:root:i=642 residual=0.0009250384173355997
    DEBUG:root:i=643 residual=0.0009192394791170955
    DEBUG:root:i=644 residual=0.000916125369258225
    DEBUG:root:i=645 residual=0.00089988176478073
    DEBUG:root:i=646 residual=0.000894081371370703
    DEBUG:root:i=647 residual=0.0008909701136872172
    DEBUG:root:i=648 residual=0.000877344748005271
    DEBUG:root:i=649 residual=0.0008716354495845735
    DEBUG:root:i=650 residual=0.0008632458047941327
    DEBUG:root:i=651 residual=0.0008470868342556059
    DEBUG:root:i=652 residual=0.0008413073373958468
    DEBUG:root:i=653 residual=0.0008356155012734234
    DEBUG:root:i=654 residual=0.0008272416889667511
    DEBUG:root:i=655 residual=0.000821547640953213
    DEBUG:root:i=656 residual=0.0008132383809424937
    DEBUG:root:i=657 residual=0.0008048746967688203
    DEBUG:root:i=658 residual=0.0007991879829205573
    DEBUG:root:i=659 residual=0.0007856738520786166
    DEBUG:root:i=660 residual=0.00078253832180053
    DEBUG:root:i=661 residual=0.0007794732809998095
    DEBUG:root:i=662 residual=0.0007607336156070232
    DEBUG:root:i=663 residual=0.0007576796924695373
    DEBUG:root:i=664 residual=0.0007571731111966074
    DEBUG:root:i=665 residual=0.0007384444470517337
    DEBUG:root:i=666 residual=0.0007353920955210924
    DEBUG:root:i=667 residual=0.000727130682207644
    DEBUG:root:i=668 residual=0.0007240860722959042
    DEBUG:root:i=669 residual=0.0007184323621913791
    DEBUG:root:i=670 residual=0.0007075525354593992
    DEBUG:root:i=671 residual=0.0007019090699031949
    DEBUG:root:i=672 residual=0.0007014945731498301
    DEBUG:root:i=673 residual=0.0006880218279547989
    DEBUG:root:i=674 residual=0.0006797607056796551
    DEBUG:root:i=675 residual=0.0006767489248886704
    DEBUG:root:i=676 residual=0.0006658813799731433
    DEBUG:root:i=677 residual=0.0006602516514249146
    DEBUG:root:i=678 residual=0.0006572349229827523
    DEBUG:root:i=679 residual=0.0006490077939815819
    DEBUG:root:i=680 residual=0.0006459152791649103
    DEBUG:root:i=681 residual=0.0006455258699133992
    DEBUG:root:i=682 residual=0.0006268468569032848
    DEBUG:root:i=683 residual=0.0006265355623327196
    DEBUG:root:i=684 residual=0.0006235238397493958
    DEBUG:root:i=685 residual=0.0006100796745158732
    DEBUG:root:i=686 residual=0.0006044654292054474
    DEBUG:root:i=687 residual=0.0005962501163594425
    DEBUG:root:i=688 residual=0.0005907180020585656
    DEBUG:root:i=689 residual=0.0005851121386513114
    DEBUG:root:i=690 residual=0.0005847280262969434
    DEBUG:root:i=691 residual=0.0005713863065466285
    DEBUG:root:i=692 residual=0.0005683911731466651
    DEBUG:root:i=693 residual=0.0005628258804790676
    DEBUG:root:i=694 residual=0.0005625305348075926
    DEBUG:root:i=695 residual=0.0005543135339394212
    DEBUG:root:i=696 residual=0.0005487268790602684
    DEBUG:root:i=697 residual=0.0005484315915964544
    DEBUG:root:i=698 residual=0.0005376291228458285
    DEBUG:root:i=699 residual=0.0005347299738787115
    DEBUG:root:i=700 residual=0.0005317716277204454
    DEBUG:root:i=701 residual=0.0005262626218609512
    DEBUG:root:i=702 residual=0.0005154639366082847
    DEBUG:root:i=703 residual=0.0005099591217003763
    DEBUG:root:i=704 residual=0.0005070339539088309
    DEBUG:root:i=705 residual=0.0005067590973339975
    DEBUG:root:i=706 residual=0.0005011943867430091
    DEBUG:root:i=707 residual=0.00048784224782139063
    DEBUG:root:i=708 residual=0.0004875008307863027
    DEBUG:root:i=709 residual=0.00048465150757692754
    DEBUG:root:i=710 residual=0.00047908781562000513
    DEBUG:root:i=711 residual=0.0004736068658530712
    DEBUG:root:i=712 residual=0.0004733417008537799
    DEBUG:root:i=713 residual=0.00046779599506407976
    DEBUG:root:i=714 residual=0.00045967803453095257
    DEBUG:root:i=715 residual=0.00045942465658299625
    DEBUG:root:i=716 residual=0.000451291271019727
    DEBUG:root:i=717 residual=0.00045103917364031076
    DEBUG:root:i=718 residual=0.00044557228102348745
    DEBUG:root:i=719 residual=0.00043739640386775136
    DEBUG:root:i=720 residual=0.00043714133789762855
    DEBUG:root:i=721 residual=0.0004316789854783565
    DEBUG:root:i=722 residual=0.00042621284956112504
    DEBUG:root:i=723 residual=0.0004206812591291964
    DEBUG:root:i=724 residual=0.0004178134840913117
    DEBUG:root:i=725 residual=0.0004097599012311548
    DEBUG:root:i=726 residual=0.00040951158734969795
    DEBUG:root:i=727 residual=0.00040406204061582685
    DEBUG:root:i=728 residual=0.0003985283838119358
    DEBUG:root:i=729 residual=0.00039830629248172045
    DEBUG:root:i=730 residual=0.0003954899148084223
    DEBUG:root:i=731 residual=0.0003952623810619116
    DEBUG:root:i=732 residual=0.0003950339159928262
    DEBUG:root:i=733 residual=0.0003791026247199625
    DEBUG:root:i=734 residual=0.00037887509097345173
    DEBUG:root:i=735 residual=0.00037859223084524274
    DEBUG:root:i=736 residual=0.00037318281829357147
    DEBUG:root:i=737 residual=0.0003729648014996201
    DEBUG:root:i=738 residual=0.00036751944571733475
    DEBUG:root:i=739 residual=0.0003620823554228991
    DEBUG:root:i=740 residual=0.0003566433151718229
    DEBUG:root:i=741 residual=0.0003563909267541021
    DEBUG:root:i=742 residual=0.000356184842530638
    DEBUG:root:i=743 residual=0.0003481399908196181
    DEBUG:root:i=744 residual=0.00034270345349796116
    DEBUG:root:i=745 residual=0.00034249667078256607
    DEBUG:root:i=746 residual=0.00033706819522194564
    DEBUG:root:i=747 residual=0.0003342406125739217
    DEBUG:root:i=748 residual=0.0003288167354185134
    DEBUG:root:i=749 residual=0.00032344015198759735
    DEBUG:root:i=750 residual=0.0003205868124496192
    DEBUG:root:i=751 residual=0.0003203896339982748
    DEBUG:root:i=752 residual=0.0003201868094038218
    DEBUG:root:i=753 residual=0.00031476846197620034
    DEBUG:root:i=754 residual=0.0003145721566397697
    DEBUG:root:i=755 residual=0.0003091557009611279
    DEBUG:root:i=756 residual=0.00030902717844583094
    DEBUG:root:i=757 residual=0.00030098945717327297
    DEBUG:root:i=758 residual=0.0003007943741977215
    DEBUG:root:i=759 residual=0.0003006096521858126
    DEBUG:root:i=760 residual=0.0002978194097522646
    DEBUG:root:i=761 residual=0.0002924067957792431
    DEBUG:root:i=762 residual=0.0002869992458727211
    DEBUG:root:i=763 residual=0.000286878552287817
    DEBUG:root:i=764 residual=0.00028669601306319237
    DEBUG:root:i=765 residual=0.0002839449734892696
    DEBUG:root:i=766 residual=0.0002837575157172978
    DEBUG:root:i=767 residual=0.0002783523523248732
    DEBUG:root:i=768 residual=0.0002728927065618336
    DEBUG:root:i=769 residual=0.0002675635041669011
    DEBUG:root:i=770 residual=0.00026738413725979626
    DEBUG:root:i=771 residual=0.0002672048285603523
    DEBUG:root:i=772 residual=0.0002670355315785855
    DEBUG:root:i=773 residual=0.0002591072116047144
    DEBUG:root:i=774 residual=0.00025892749545164406
    DEBUG:root:i=775 residual=0.000256130238994956
    DEBUG:root:i=776 residual=0.0002507370081730187
    DEBUG:root:i=777 residual=0.0002506973105482757
    DEBUG:root:i=778 residual=0.00024001835845410824
    DEBUG:root:i=779 residual=0.00023991518537513912
    DEBUG:root:i=780 residual=0.00023974555369932204
    DEBUG:root:i=781 residual=0.0002396399067947641
    DEBUG:root:i=782 residual=0.00023947634326759726
    DEBUG:root:i=783 residual=0.00023665565822739154
    DEBUG:root:i=784 residual=0.00023133648210205138
    DEBUG:root:i=785 residual=0.0002311761782038957
    DEBUG:root:i=786 residual=0.00023101236729416996
    DEBUG:root:i=787 residual=0.0002309219999006018
    DEBUG:root:i=788 residual=0.0002255312429042533
    DEBUG:root:i=789 residual=0.00022015496506355703
    DEBUG:root:i=790 residual=0.00022006740618962795
    DEBUG:root:i=791 residual=0.00021990612731315196
    DEBUG:root:i=792 residual=0.000219747846131213
    DEBUG:root:i=793 residual=0.00020921867690049112
    DEBUG:root:i=794 residual=0.00020905987184960395
    DEBUG:root:i=795 residual=0.00020896551723126322
    DEBUG:root:i=796 residual=0.00020881620002910495
    DEBUG:root:i=797 residual=0.0002086693566525355
    DEBUG:root:i=798 residual=0.00020859223150182515
    DEBUG:root:i=799 residual=0.0002032057527685538
    DEBUG:root:i=800 residual=0.00020313114509917796
    DEBUG:root:i=801 residual=0.0002029876341111958
    DEBUG:root:i=802 residual=0.00019761397561524063
    DEBUG:root:i=803 residual=0.00019229803001508117
    DEBUG:root:i=804 residual=0.0001921611838042736
    DEBUG:root:i=805 residual=0.00018944544717669487
    DEBUG:root:i=806 residual=0.00018930304213427007
    DEBUG:root:i=807 residual=0.00018922315211966634
    DEBUG:root:i=808 residual=0.00018908835772890598
    DEBUG:root:i=809 residual=0.00018901756266131997
    DEBUG:root:i=810 residual=0.00018888473277911544
    DEBUG:root:i=811 residual=0.00018090980302076787
    DEBUG:root:i=812 residual=0.00018343272677157074
    DEBUG:root:i=813 residual=0.00017025298438966274
    DEBUG:root:i=814 residual=0.00016755603428464383
    DEBUG:root:i=815 residual=0.00016740718274377286
    DEBUG:root:i=816 residual=0.00016733963275328279
    DEBUG:root:i=817 residual=0.00016719500126782805
    DEBUG:root:i=818 residual=0.0001671310019446537
    DEBUG:root:i=819 residual=0.00016177172074094415
    DEBUG:root:i=820 residual=0.00015910818183328956
    DEBUG:root:i=821 residual=0.00016156271158251911
    DEBUG:root:i=822 residual=0.00015891021757852286
    DEBUG:root:i=823 residual=0.0001588449813425541
    DEBUG:root:i=824 residual=0.00015086277562659234
    DEBUG:root:i=825 residual=0.0001507946290075779
    DEBUG:root:i=826 residual=0.00014808113337494433
    DEBUG:root:i=827 residual=0.00014801554789301008
    DEBUG:root:i=828 residual=0.00015046718181110919
    DEBUG:root:i=829 residual=0.0001478140038670972
    DEBUG:root:i=830 residual=0.00014769233530387282
    DEBUG:root:i=831 residual=0.00015021204308141023
    DEBUG:root:i=832 residual=0.00014750090485904366
    DEBUG:root:i=833 residual=0.00014480657409876585
    DEBUG:root:i=834 residual=0.0001447473478037864
    DEBUG:root:i=835 residual=0.0001446283422410488
    DEBUG:root:i=836 residual=0.00014455527707468718
    DEBUG:root:i=837 residual=0.00013398843293543905
    DEBUG:root:i=838 residual=0.00013393332483246922
    DEBUG:root:i=839 residual=0.00013387762010097504
    DEBUG:root:i=840 residual=0.00012596829037647694
    DEBUG:root:i=841 residual=0.00012848532060161233
    DEBUG:root:i=842 residual=0.00012577655434142798
    DEBUG:root:i=843 residual=0.00012571994739118963
    DEBUG:root:i=844 residual=0.00012824776058550924
    DEBUG:root:i=845 residual=0.00012553902342915535
    DEBUG:root:i=846 residual=0.00012548876111395657
    DEBUG:root:i=847 residual=0.0001280204887734726
    DEBUG:root:i=848 residual=0.0001253146620001644
    DEBUG:root:i=849 residual=0.00012262102973181754
    DEBUG:root:i=850 residual=0.0001225066080223769
    DEBUG:root:i=851 residual=0.00012245945981703699
    DEBUG:root:i=852 residual=0.00011718744644895196
    DEBUG:root:i=853 residual=0.00011971580534009263
    DEBUG:root:i=854 residual=0.00011701765470206738
    DEBUG:root:i=855 residual=0.00011961082782363519
    DEBUG:root:i=856 residual=0.00011950052430620417
    DEBUG:root:i=857 residual=0.00011681034084176645
    DEBUG:root:i=858 residual=0.00011941552656935528
    DEBUG:root:i=859 residual=0.00010621196997817606
    DEBUG:root:i=860 residual=0.0001088065910153091
    DEBUG:root:i=861 residual=0.0001061112925526686
    DEBUG:root:i=862 residual=0.00010864000068977475
    DEBUG:root:i=863 residual=0.00010859588655875996
    DEBUG:root:i=864 residual=0.00010590952297206968
    DEBUG:root:i=865 residual=0.0001084494506358169
    DEBUG:root:i=866 residual=0.00010840308823389933
    DEBUG:root:i=867 residual=0.00010571617895038798
    DEBUG:root:i=868 residual=0.00010826283687492833
    DEBUG:root:i=869 residual=0.00010821739851962775
    DEBUG:root:i=870 residual=9.772025077836588e-05
    DEBUG:root:i=871 residual=0.00010025152005255222
    DEBUG:root:i=872 residual=0.00010014374856837094
    DEBUG:root:i=873 residual=0.00010010979167418554
    DEBUG:root:i=874 residual=0.00010007217497332022
    DEBUG:root:i=875 residual=9.996776498155668e-05
    DEBUG:root:i=876 residual=9.734965715324506e-05
    DEBUG:root:i=877 residual=9.465336916036904e-05
    DEBUG:root:i=878 residual=9.72716516116634e-05
    DEBUG:root:i=879 residual=9.19386075111106e-05
    DEBUG:root:i=880 residual=9.190053242491558e-05
    DEBUG:root:i=881 residual=8.921598055167124e-05
    DEBUG:root:i=882 residual=8.661368337925524e-05
    DEBUG:root:i=883 residual=8.385876571992412e-05
    DEBUG:root:i=884 residual=8.647540380479768e-05
    DEBUG:root:i=885 residual=8.643559704069048e-05
    DEBUG:root:i=886 residual=8.367257396457717e-05
    DEBUG:root:i=887 residual=8.629429066786543e-05
    DEBUG:root:i=888 residual=8.10435158200562e-05
    DEBUG:root:i=889 residual=8.100664854282513e-05
    DEBUG:root:i=890 residual=7.825560896890238e-05
    DEBUG:root:i=891 residual=7.302505400730297e-05
    DEBUG:root:i=892 residual=7.56281369831413e-05
    DEBUG:root:i=893 residual=7.295318209799007e-05
    DEBUG:root:i=894 residual=7.556080527137965e-05
    DEBUG:root:i=895 residual=7.544810068793595e-05
    DEBUG:root:i=896 residual=7.022321369731799e-05
    DEBUG:root:i=897 residual=7.018883479759097e-05
    DEBUG:root:i=898 residual=7.26986036170274e-05
    DEBUG:root:i=899 residual=7.004607323324308e-05
    DEBUG:root:i=900 residual=7.001439371379092e-05
    DEBUG:root:i=901 residual=7.253673538798466e-05
    DEBUG:root:i=902 residual=6.995238072704524e-05
    DEBUG:root:i=903 residual=6.99229640304111e-05
    DEBUG:root:i=904 residual=7.238053512992337e-05
    DEBUG:root:i=905 residual=6.979449244681746e-05
    DEBUG:root:i=906 residual=6.975758878979832e-05
    DEBUG:root:i=907 residual=7.229319453472272e-05
    DEBUG:root:i=908 residual=6.963524356251583e-05
    DEBUG:root:i=909 residual=6.961033795960248e-05
    DEBUG:root:i=910 residual=7.21478063496761e-05
    DEBUG:root:i=911 residual=6.955662684049457e-05
    DEBUG:root:i=912 residual=6.686279084533453e-05
    DEBUG:root:i=913 residual=6.67689528199844e-05
    DEBUG:root:i=914 residual=6.674423639196903e-05
    DEBUG:root:i=915 residual=6.672021117992699e-05
    DEBUG:root:i=916 residual=6.669331924058497e-05
    DEBUG:root:i=917 residual=6.666353874607012e-05
    DEBUG:root:i=918 residual=6.14138989476487e-05
    DEBUG:root:i=919 residual=5.8773184719029814e-05
    DEBUG:root:i=920 residual=6.129351095296443e-05
    DEBUG:root:i=921 residual=5.8724373957375064e-05
    DEBUG:root:i=922 residual=5.869671076652594e-05
    DEBUG:root:i=923 residual=6.122098420746624e-05
    DEBUG:root:i=924 residual=5.858225267729722e-05
    DEBUG:root:i=925 residual=5.855612107552588e-05
    DEBUG:root:i=926 residual=6.108998059062287e-05
    DEBUG:root:i=927 residual=5.330592830432579e-05
    DEBUG:root:i=928 residual=5.581489676842466e-05
    DEBUG:root:i=929 residual=5.579267235589214e-05
    DEBUG:root:i=930 residual=5.569763743551448e-05
    DEBUG:root:i=931 residual=4.795238055521622e-05
    DEBUG:root:i=932 residual=5.043875353294425e-05
    DEBUG:root:i=933 residual=4.790018283529207e-05
    DEBUG:root:i=934 residual=4.787759462487884e-05
    DEBUG:root:i=935 residual=5.03714763908647e-05
    DEBUG:root:i=936 residual=4.7753521357662976e-05
    DEBUG:root:i=937 residual=4.7731795348227024e-05
    DEBUG:root:i=938 residual=5.02366638102103e-05
    DEBUG:root:i=939 residual=4.768930375576019e-05
    DEBUG:root:i=940 residual=4.766554411617108e-05
    DEBUG:root:i=941 residual=5.017451258026995e-05
    DEBUG:root:i=942 residual=4.7624966100556776e-05
    DEBUG:root:i=943 residual=4.7536777856294066e-05
    DEBUG:root:i=944 residual=5.005067941965535e-05
    DEBUG:root:i=945 residual=4.748839637613855e-05
    DEBUG:root:i=946 residual=4.746962440549396e-05
    DEBUG:root:i=947 residual=4.999098746338859e-05
    DEBUG:root:i=948 residual=4.7433190047740936e-05
    DEBUG:root:i=949 residual=4.7415953304152936e-05
    DEBUG:root:i=950 residual=4.987398278899491e-05
    DEBUG:root:i=951 residual=4.731310036731884e-05
    DEBUG:root:i=952 residual=4.729679130832665e-05
    DEBUG:root:i=953 residual=4.2067891627084464e-05
    DEBUG:root:i=954 residual=3.937091969419271e-05
    DEBUG:root:i=955 residual=3.686713898787275e-05
    DEBUG:root:i=956 residual=3.932859908672981e-05
    DEBUG:root:i=957 residual=3.674978142953478e-05
    DEBUG:root:i=958 residual=3.673108949442394e-05
    DEBUG:root:i=959 residual=3.920222297892906e-05
    DEBUG:root:i=960 residual=3.6693971196655184e-05
    DEBUG:root:i=961 residual=3.66746389772743e-05
    DEBUG:root:i=962 residual=3.9147686038631946e-05
    DEBUG:root:i=963 residual=3.662988092401065e-05
    DEBUG:root:i=964 residual=3.661472874227911e-05
    DEBUG:root:i=965 residual=3.9030615880619735e-05
    DEBUG:root:i=966 residual=3.6511952202999964e-05
    DEBUG:root:i=967 residual=3.64943734894041e-05
    DEBUG:root:i=968 residual=3.898487193509936e-05
    DEBUG:root:i=969 residual=3.6461213312577456e-05
    DEBUG:root:i=970 residual=3.644679600256495e-05
    DEBUG:root:i=971 residual=3.894007750204764e-05
    DEBUG:root:i=972 residual=3.6416207876754925e-05
    DEBUG:root:i=973 residual=3.6400309909367934e-05
    DEBUG:root:i=974 residual=3.883795579895377e-05
    DEBUG:root:i=975 residual=3.630630089901388e-05
    DEBUG:root:i=976 residual=3.62953869625926e-05
    DEBUG:root:i=977 residual=3.879847645293921e-05
    DEBUG:root:i=978 residual=3.626613397500478e-05
    DEBUG:root:i=979 residual=3.6251476558391005e-05
    DEBUG:root:i=980 residual=3.876052869600244e-05
    DEBUG:root:i=981 residual=3.6225821531843394e-05
    DEBUG:root:i=982 residual=3.62127429980319e-05
    DEBUG:root:i=983 residual=3.3487096516182646e-05
    DEBUG:root:i=984 residual=3.340421608299948e-05
    DEBUG:root:i=985 residual=3.339362592669204e-05
    DEBUG:root:i=986 residual=3.338011083542369e-05
    DEBUG:root:i=987 residual=3.336870940984227e-05
    DEBUG:root:i=988 residual=3.335567089379765e-05
    DEBUG:root:i=989 residual=3.3344658731948584e-05
    DEBUG:root:i=990 residual=3.333346467115916e-05
    DEBUG:root:i=991 residual=3.3320917282253504e-05
    DEBUG:root:i=992 residual=2.8098564143874682e-05
    DEBUG:root:i=993 residual=3.08047492580954e-05
    DEBUG:root:i=994 residual=3.079557791352272e-05
    DEBUG:root:i=995 residual=2.799366120598279e-05
    DEBUG:root:i=996 residual=3.07085138047114e-05
    DEBUG:root:i=997 residual=3.0697687179781497e-05
    DEBUG:root:i=998 residual=2.7957106794929132e-05
    DEBUG:root:i=999 residual=3.067626676056534e-05
    INFO:root:rank=0 pagerank=4.7948e+01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
    INFO:root:rank=1 pagerank=4.7948e+01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
    INFO:root:rank=2 pagerank=7.2710e+00 url=www.lawfareblog.com/cost-using-zero-days
    INFO:root:rank=3 pagerank=2.1691e+00 url=www.lawfareblog.com/lawfare-podcast-former-congressman-brian-baird-and-daniel-schuman-how-congress-can-continue-function
    INFO:root:rank=4 pagerank=1.4214e+00 url=www.lawfareblog.com/events
    INFO:root:rank=5 pagerank=1.0862e+00 url=www.lawfareblog.com/water-wars-increased-us-focus-indo-pacific
    INFO:root:rank=6 pagerank=1.0862e+00 url=www.lawfareblog.com/water-wars-drill-maybe-drill
    INFO:root:rank=7 pagerank=1.0862e+00 url=www.lawfareblog.com/water-wars-disjointed-operations-south-china-sea
    INFO:root:rank=8 pagerank=1.0862e+00 url=www.lawfareblog.com/water-wars-us-china-divide-shangri-la
    INFO:root:rank=9 pagerank=1.0862e+00 url=www.lawfareblog.com/water-wars-sinking-feeling-philippine-china-relations
   ```

   Task 2, part 1:
   ```
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='corona'
    INFO:root:rank=0 pagerank=8.8870e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
    INFO:root:rank=1 pagerank=8.8867e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
    INFO:root:rank=2 pagerank=1.8256e-01 url=www.lawfareblog.com/chinatalk-how-party-takes-its-propaganda-global
    INFO:root:rank=3 pagerank=1.4907e-01 url=www.lawfareblog.com/rational-security-my-corona-edition
    INFO:root:rank=4 pagerank=1.4907e-01 url=www.lawfareblog.com/brexit-not-immune-coronavirus
    INFO:root:rank=5 pagerank=1.0729e-01 url=www.lawfareblog.com/trump-cant-reopen-country-over-state-objections
    INFO:root:rank=6 pagerank=1.0199e-01 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
    INFO:root:rank=7 pagerank=1.0199e-01 url=www.lawfareblog.com/britains-coronavirus-response
    INFO:root:rank=8 pagerank=9.4298e-02 url=www.lawfareblog.com/lawfare-podcast-mom-and-dad-talk-clinical-trials-pandemic
    INFO:root:rank=9 pagerank=8.7207e-02 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
   ```

   Task 2, part 2:
   ```
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='corona' --search_query='-corona'
    INFO:root:rank=0 pagerank=8.8870e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
    INFO:root:rank=1 pagerank=8.8867e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
    INFO:root:rank=2 pagerank=1.8256e-01 url=www.lawfareblog.com/chinatalk-how-party-takes-its-propaganda-global
    INFO:root:rank=3 pagerank=1.0729e-01 url=www.lawfareblog.com/trump-cant-reopen-country-over-state-objections
    INFO:root:rank=4 pagerank=9.4298e-02 url=www.lawfareblog.com/lawfare-podcast-mom-and-dad-talk-clinical-trials-pandemic
    INFO:root:rank=5 pagerank=7.9633e-02 url=www.lawfareblog.com/fault-lines-foreign-policy-quarantined
    INFO:root:rank=6 pagerank=7.5307e-02 url=www.lawfareblog.com/limits-world-health-organization
    INFO:root:rank=7 pagerank=6.8115e-02 url=www.lawfareblog.com/chinatalk-dispatches-shanghai-beijing-and-hong-kong
    INFO:root:rank=8 pagerank=6.4847e-02 url=www.lawfareblog.com/us-moves-dismiss-case-against-company-linked-ira-troll-farm
    INFO:root:rank=9 pagerank=6.4847e-02 url=www.lawfareblog.com/livestream-house-armed-services-committee-holds-hearing-priorities-missile-defense
   ```

1. Ensure that all your changes to the `pagerank.py` and `README.md` files are committed to your repo and pushed to github.

1. Get at least 5 stars on your repo.
   (You may trade stars with other students in the class.)

   > **NOTE:**
   > 
   > Recruiters use github profiles to determine who to hire,
   > and pagerank is used to rank user profiles and projects.
   > Links in this graph correspond to who has starred/followed who's repo.
   > By getting more stars on your repo, you'll be increasing your github pagerank, which increases the likelihood that recruiters will hire you.
   > To see an example, [perform a search for `data mining`](https://github.com/search?q=data+mining).
   > Notice that the results are returned "approximately" ranked by the number of stars,
   > but because "some stars count more than others" the results are not exactly ranked by the number of stars.
   > (I asked you not to fork this repo because forks are ranked lower than non-forks.)
   >
   > In some sense, we are doing a "dual problem" to data mining by getting these stars.
   > Recruiters are using data mining to find out who the best people to recruit are,
   > and we are hacking their data mining algorithms by making those algorithms select you instead of someone else.
   >
   > If you're interested in exploring this idea further, here's a python tutorial for extracting GitHub's social graph: <https://www.oreilly.com/library/view/mining-the-social/9781449368180/ch07.html> ; if you're interested in learning more about how recruiters use github profiles, read this Hacker News post: <https://news.ycombinator.com/item?id=19413348>.

1. Submit the url of your repo to sakai.

   The assignment is worth 8 points.
   1. There are 6 parts to the output above.  (4 in Task1 and 2 in Task2.)
   1. Each part that you get incorrect will result in -2 points.  (But you cannot go negative.)
   1. Another way of phrasing this is that the first 2 parts you complete are not worth any points,
      but each part after that is worth 2 points.
