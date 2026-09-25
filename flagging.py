# Flagging logic, decide if the toxicity/setiment score should be toxic or not

def baseline(toxicity, sentiment):
    #current rule used, only uses toxicity score
    return toxicity > 0.7


def option_a(toxicity, sentiment):
    #same as baseline, but flags if toxicity is medium (> 0.50 AND sentiment is negative (<-0.5)
    #try to cathch borderline toxic clips that are clearly hostile
    return toxicity > 0.7 or (toxicity > 0.5 and sentiment < -0.5)


def option_b(toxicity, sentiment):
    #stricter, needs both toxicity (>0.6) and negatve sentiment(<-0.3)
    return toxicity > 0.6 and sentiment < -0.3


#all options together so files can loop through them
FLAGGING_OPTIONS = {
    "baseline": baseline,
    "option_a": option_a,
    "option_b": option_b,
}