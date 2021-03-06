\name{find.closest}
\alias{find.closest}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
	FIND CLOSEST CLUSTERS
}
\description{
	Determines which two segments to merge.
}
\usage{
find.closest(K, ret)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{K}{Integer indicating the progress of the agglomerative process.}
  \item{ret}{A list with 'open', 'N', and 'right' components}
}
\details{
This method is called by the e.agglomerative method, and should not be called 
by the user.
%%  ~~ If necessary, more details than the description above ~~
}
\value{
Returns a vector with 3 components. The first two indicate which segments 
are to be merged. The third is the new goodness of fit statistics.
%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
}
\references{
%% ~put references to the literature/web site here ~
James NA, Matteson DS (2013). A Nonparametric Approach for Multiple Change Point Analysis of Multivariate Data.

Nicholas A. James, David S. Matteson (2014). "ecp: An R Package for Nonparametric 
Multiple Change Point Analysis of Multivariate Data.", "Journal of Statistical Software, 
62(7), 1-25", URL "http://www.jstatsoft.org/v62/i07/"

Rizzo ML, Szekely GL (2005). Hierarchical clustering via joint between-within distances: Extending ward's minimum variance method. Journal of Classification. pp. 151 - 183.
}
\author{
Nicholas A. James
}


%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
\code{\link{e.agglo}}
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ internal }
