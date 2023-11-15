import numpy as np


def joint_distribution_of_word_counts(texts, word0, word1):
    """
    Parameters:
    texts (list of lists) - a list of texts; each text is a list of words
    word0 (str) - the first word to count
    word1 (str) - the second word to count

    Output:
    Pjoint (numpy array) - Pjoint[m,n] = P(X0=m,X1=n), where
      X0 is the number of times that word0 occurs in a given text,
      X1 is the number of times that word1 occurs in the same text.
    """
    # 단어 카운트를 위한 최대값 초기화
    max_count_word0 = 0
    max_count_word1 = 0
    
    # 각 텍스트에서 단어의 카운트를 저장하기 위한 리스트
    counts_word0 = []
    counts_word1 = []

    # 모든 텍스트에 대해 단어 카운트
    for text in texts:
        count_word0 = text.count(word0)
        count_word1 = text.count(word1)

        counts_word0.append(count_word0)
        counts_word1.append(count_word1)

        # 최대 카운트 업데이트
        if count_word0 > max_count_word0:
            max_count_word0 = count_word0
        if count_word1 > max_count_word1:
            max_count_word1 = count_word1

    # Pjoint 배열 초기화
    Pjoint = np.zeros((max_count_word0 + 1, max_count_word1 + 1))

    # 결합 분포 계산
    for count_word0, count_word1 in zip(counts_word0, counts_word1):
        Pjoint[count_word0, count_word1] += 1

    # 정규화
    Pjoint /= np.sum(Pjoint)

    return Pjoint



def marginal_distribution_of_word_counts(Pjoint, index):
    """
    Parameters:
    Pjoint (numpy array) - Pjoint[m,n] = P(X0=m,X1=n), where
      X0 is the number of times that word0 occurs in a given text,
      X1 is the number of times that word1 occurs in the same text.
    index (0 or 1) - which variable to retain (marginalize the other)

    Output:
    Pmarginal (numpy array) - Pmarginal[x] = P(X=x), where
      if index==0, then X is X0
      if index==1, then X is X1
    """
    # 주변 확률 분포 계산
    if index == 0:
        # index가 0이면, 각 행의 합을 계산
        Pmarginal = np.sum(Pjoint, axis=1)
    elif index == 1:
        # index가 1이면, 각 열의 합을 계산
        Pmarginal = np.sum(Pjoint, axis=0)
    else:
        # 잘못된 인덱스 입력 처리
        raise ValueError("Index must be 0 or 1")

    return Pmarginal


def conditional_distribution_of_word_counts(Pjoint, Pmarginal):
    """
    Parameters:
    Pjoint (numpy array) - Pjoint[m,n] = P(X0=m,X1=n), where
      X0 is the number of times that word0 occurs in a given text,
      X1 is the number of times that word1 occurs in the same text.
    Pmarginal (numpy array) - Pmarginal[m] = P(X0=m)

    Outputs:
    Pcond (numpy array) - Pcond[m,n] = P(X1=n|X0=m)
    """
    # Pcond 배열 초기화
    Pcond = np.zeros_like(Pjoint)

    # 조건부 확률 계산
    for m in range(Pjoint.shape[0]):
        for n in range(Pjoint.shape[1]):
            if Pmarginal[m] > 0:
                Pcond[m, n] = Pjoint[m, n] / Pmarginal[m]
            else:
                # Pmarginal[m]이 0인 경우 Pcond[m, n]을 NaN으로 설정
                Pcond[m, n] = np.nan

    return Pcond

def mean_from_distribution(P):
    """
    Parameters:
    P (numpy array) - P[n] = P(X=n)

    Outputs:
    mu (float) - the mean of X
    """
    # 확률 분포와 인덱스를 곱하여 기대값을 계산
    n = np.arange(len(P))
    mu = round(np.sum(n * P), 3)
    
    return mu


def variance_from_distribution(P):
    """
    Parameters:
    P (numpy array) - P[n] = P(X=n)

    Outputs:
    var (float) - the variance of X
    """
    # 평균 계산
    n = np.arange(len(P))
    mu = np.sum(n * P)
    
    # 분산 계산 (기대값의 제곱의 평균 - 평균의 제곱)
    var = round(np.sum((n**2) * P) - mu**2, 3)
    
    return var


def covariance_from_distribution(P):
    """
    Parameters:
    P (numpy array) - P[m,n] = P(X0=m,X1=n)

    Outputs:
    covar (float) - the covariance of X0 and X1
    """
    # 각 변수에 대한 평균 계산
    m, n = np.indices(P.shape)
    mu_m = np.sum(m * P)
    mu_n = np.sum(n * P)
    
    # 공분산 계산
    covar = round(np.sum((m - mu_m) * (n - mu_n) * P), 3)
    
    return covar

def expectation_of_a_function(P, f):
    """
    Parameters:
    P (numpy array) - joint distribution, P[m,n] = P(X0=m,X1=n)
    f (function) - f should be a function that takes two
       real-valued inputs, x0 and x1.  The output, z=f(x0,x1),
       must be a real number for all values of (x0,x1)
       such that P(X0=x0,X1=x1) is nonzero.

    Output:
    expected (float) - the expected value, E[f(X0,X1)]
    """
    m, n = np.indices(P.shape)
    expected = round(np.sum(f(m, n) * P), 3)
    
    return expected
