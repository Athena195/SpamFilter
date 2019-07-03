import os # Modul os có các hàm làm việc với file và folder
from pyvi import ViTokenizer # Thư viện xử lý tách từ trong Tiếng Việt
from gensim.parsing.preprocessing import strip_non_alphanum, strip_multiple_whitespaces,preprocess_string, split_alphanum, strip_short, strip_numeric # Thư viện
import re # Thư viện xử lý các văn bản
import numpy as np # Thư viện để xử lý số liệu, tính toán
import math

# Tiền xử lý mail
def preprocess_mail(mail):
    mail = re.sub("http\S+", "", mail) # Loại bỏ các đường dẫn
    mail = strip_non_alphanum(mail).lower().strip() # Loại bỏ các kí tự không phải là chữ cái, chuyển tất cả kí tự thành chữ thường
    mail = split_alphanum(mail) # Tách văn bản thành các từ
    mail = strip_short(mail, minsize=2) # Lấy các từ có độ dài >= 2 kí tự, loại bỏ các từ có 1 chữ cái
    mail = strip_numeric(mail) #
    mail = ViTokenizer.tokenize(mail)
    return mail

# Hàm so sánh xác suất mail là spam hoặc là nonspam
def compare(predict_spam, predict_non_spam):
    if predict_spam > predict_non_spam:
        return True
    return False

# Hàm dự đoán mail là spam hay là nonspam
def predict_mail(bag_Of_Words, bayes_matrix, test_mail):
    label = 0
    predict_spam = 0
    predict_non_spam = 0
    m = open(test_mail, encoding='utf8')
    words = []
    for i, line in enumerate(m):  # hàm enumerate() cho phép bạn truy nhập vòng lặp lần lượt qua các thành phần của một collection trong khi nó vẫn giữ index của item hiện tại.
        line = preprocess_mail(line)
        word = line.split()
        words += word
    #print(words)
    vector = np.zeros(len(bag_Of_Words))
    for i, word in enumerate(bag_Of_Words):
        if word in words:
            vector[i] = 1
    for i, v in enumerate(vector):
        if v == 0:
            predict_spam += bayes_matrix[i][2]
            predict_non_spam += bayes_matrix[i][3]
        else:
            predict_spam += bayes_matrix[i][0]
            predict_non_spam += bayes_matrix[i][1]
    #print(predict_spam, " ", predict_non_spam)
    if compare(predict_spam, predict_non_spam):
        label = 1
    return label
#
def predict(test_spam,bayes_matrix, bag_Of_Words):
    labels = []
    test_spam.encode('utf8')
    emails = os.listdir(test_spam)  # danh sách các tên file nằm trong thư mục train_spam
    for mail in emails:
        mail = os.path.join(test_spam,mail)  # os.path.join nhận vào một thư mục và một tên file rồi nối chúng lại thành một đường dẫn hoàn chỉnh.
        labels.append(predict_mail(bag_Of_Words, bayes_matrix, mail))
    return labels
test_spam = 'C:\\Users\\Admin\\PycharmProjects\\demo\\venv\\Test\\Test_spam'
test_nonspam = 'C:\\Users\\Admin\\PycharmProjects\\demo\\venv\\Test\\Test_nonspam'

# Lấy túi từ từ trong file
file = open('bagofwords.txt', mode='r', encoding='utf8')
bag_Of_Words = file.readline()
bag_Of_Words = bag_Of_Words.split()
file.close()

# Lấy bayes_matrix từ trong file

file = open('bayes_matrix.txt', mode='r')
n = file.readline()
n = int(n)
bayes_matrix = np.zeros((n,4))

for i in range(n):
    vector = []
    line = file.readline()
    line = line.split()
    for j, w in enumerate(line):
        w = float(w)
        bayes_matrix[i][j] = w
file.close()
#Dự đoán cho từng mail
labels = predict(test_spam,bayes_matrix,bag_Of_Words)
labels = labels + predict(test_nonspam,bayes_matrix,bag_Of_Words)
count = 0
for i, label in enumerate(labels):
    if( i<50 ):
        if( label == 1 ):
            count += 1
    else:
        if( label == 0 ):
            count += 1
print(count)




