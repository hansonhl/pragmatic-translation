# Pragmatic-Image-Captioning

This codebase implements Bayesian pragmatics (i.e. the Rational Speech Acts model - RSA) over the top of a deep neural image
captioning model. These are desirable to combine, since RSA gives rise to linguistically realistic effects, while deep models can capture (at least some) of the flexibility and expressivity of natural language.

Very short summary below, but see notebooks/main.ipynb for a more in-depth overview, or use "ipython main.py" to run the model with some preselected images and a pretrained neural model:

* Suppose we have a space of possible sentences U
	
* Choosing the sentence which is the most informative caption for identifying image w out of a set of images W is a useful task (moreover, it represents a key instance of natural language pragmatics)

* Viewed as an inference problem (of a speaker agent P(U|W=w) ), this task is intractable when U is large.

* But if the space of possible sentences U is recursively generated, there's a solution: at each stage of the recursive generation of a sentence u, we perform a local inference as to the most informative next step

* Category theoretic perspective (very roughly): this amounts to mapping the inference into the coalgebra of the anamorphism used to generate the distribution over U

* Linguistic perspective (very broadly): we're pushing pragmatics into the lower levels of language, rather than adding it on top

* Computational perspective (very ambitiously): this provides us a way to get the power of Bayesian models of pragmatics (see Rational Speech Acts) with deep machine learning models powerful enough to model natural language



Setup:

	To run the model:

		requirements:

			-- python3
			-- jupyter (if you want to use the notebook)





