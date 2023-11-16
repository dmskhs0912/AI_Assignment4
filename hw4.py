import numpy as np


def joint_distribution_of_word_counts(texts, word0, word1):
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

    # Pjoint[m, n] 모든 m, n에 대해 경우의 수 계산
    for count_word0, count_word1 in zip(counts_word0, counts_word1):
        Pjoint[count_word0, count_word1] += 1

    # 전체 경우의 수로 나눠 확률 계산
    Pjoint /= np.sum(Pjoint)

    return Pjoint



def marginal_distribution_of_word_counts(Pjoint, index):
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
    # Pcond 배열 Pjoint와 동일한 형태로 초기화.
    Pcond = np.zeros_like(Pjoint)

    # 조건부 확률 계산
    for m in range(Pjoint.shape[0]): # 행의 개수만큼 반복
        for n in range(Pjoint.shape[1]): # 열의 개수만큼 반복
            if Pmarginal[m] > 0:
                Pcond[m, n] = Pjoint[m, n] / Pmarginal[m]
            else:
                # 분모가 0인 경우 Pcond[m, n]을 NaN으로 설정
                Pcond[m, n] = np.nan

    return Pcond

def mean_from_distribution(P):

    # P의 원소 개수 k개일 때, n = [0, 1, 2, ..., k-1]
    n = np.arange(len(P)) 
    # n, P 두 배열 각각 원소별로 곱해 더한 후 반올림
    mu = round(np.sum(n * P), 3)
    
    return mu


def variance_from_distribution(P):
    # 평균 계산
    n = np.arange(len(P))
    mu = np.sum(n * P)
    
    # 분산 계산 (기대값의 제곱의 평균 - 평균의 제곱)
    var = round(np.sum((n**2) * P) - mu**2, 3)
    
    return var


def covariance_from_distribution(P):
    # P의 행, 열 별 인덱스 배열을 구하고 m의 평균, n의 평균 구함
    m, n = np.indices(P.shape)
    mu_m = np.sum(m * P)
    mu_n = np.sum(n * P)

    
    # 공분산 계산
    covar = round(np.sum((m - mu_m) * (n - mu_n) * P), 3)
    
    return covar

def expectation_of_a_function(P, f):
    # P의 행, 열 별 인덱스 배열을 구함
    m, n = np.indices(P.shape)
    # f(m,n) 값에 대해 평균 계산
    expected = round(np.sum(f(m, n) * P), 3)
    
    return expected
