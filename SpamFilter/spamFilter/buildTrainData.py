import os # Modul os có các hàm làm việc với file và folder
from pyvi import ViTokenizer # Thư viện xử lý tách từ trong Tiếng Việt
from gensim.parsing.preprocessing import strip_non_alphanum, strip_multiple_whitespaces,preprocess_string, split_alphanum, strip_short, strip_numeric # Thư viện
import re # Thư viện xử lý các văn bản
import numpy as np # Thư viện để xử lý số liệu, tính toán
import math

# Tiền xử lý văn mail
def preprocess_mail(mail):
    mail = re.sub("https\S+", "", mail) # Loại bỏ các đường dẫn
    mail = strip_non_alphanum(mail).lower().strip() # Loại bỏ các kí tự không phải là chữ cái, chuyển tất cả kí tự thành chữ thường
    mail = split_alphanum(mail) # Tách văn bản thành các từ
    mail = strip_short(mail, minsize=2) # Lấy các từ có độ dài >= 2 kí tự, loại bỏ các từ có 1 chữ cái
    mail = strip_numeric(mail) #
    mail = ViTokenizer.tokenize(mail)
    return mail

# Hàm làm trơn
def smoothing(a,b):
    return math.log10((a+1)/(b+1))

# Hàm tạo túi từ
def make_Bag_Of_Words(train_spam):
    set_words = []
    train_spam.encode('utf8')
    emails = os.listdir(train_spam) #danh sách các tên file nằm trong thư mục train_spam
    for mail in emails:
        mail = os.path.join(train_spam,mail) #os.path.join nhận vào một thư mục và một tên file rồi nối chúng lại thành một đường dẫn hoàn chỉnh.
        with open(mail, encoding= 'utf8') as m: # mở file tiếng việt
            for i,line in enumerate(m): #hàm enumerate() cho phép bạn truy nhập vòng lặp lần lượt qua các thành phần của một collection trong khi nó vẫn giữ index của item hiện tại.
                line = preprocess_mail(line)
                words = line.split()
                set_words += words
    print(len(set_words))
    return set_words

# Hàm trích xuất các đặc trưng của mẫu
def extract_features(train_spam, bag_Of_Words):
    vectors = []
    train_spam.encode('utf8')
    emails = os.listdir(train_spam)
    for mail in emails:
        mail = os.path.join(train_spam,mail)
        set_words = []
        vector = np.zeros(len(bag_Of_Words)) #Khởi tạo vector từ có kích thước bằng kích thước của túi từ
        with open(mail, encoding= 'utf8') as m:
            for i,line in enumerate(m):
                line = preprocess_mail(line)
                words = line.split()
                set_words += words
            for i, words in enumerate(bag_Of_Words):
                if(words in set_words):
                    vector[i] = 1
        vectors.append(vector)
    return vectors

# Xây dụng ma trận xác suất
def build_matrix(train_spam, train_nonspam, bag_Of_Words):
    bayes_matrix = np.zeros((len(bag_Of_Words),4))  # tạo ma trận có sô hàng là kích thước của túi từ và 4 cột chứa xác xuất của 4 loại app_spam, app_nonspam, nonapp_spam, nonapp_nonsp
    vetors = extract_features(train_spam, bag_Of_Words)
    vetors = vetors + extract_features(train_nonspam, bag_Of_Words)
    for i, word in enumerate(bag_Of_Words):
        app_spam = 0
        app_nonspam = 0
        nonapp_spam = 0
        nonapp_nonspam = 0
        for k, v in enumerate(vetors):
            if v[i] == 1:
                if label[k] == 1:
                    app_spam += 1
                else:
                    app_nonspam += 1
            else:
                if label[k] == 1:
                    nonapp_spam += 1
                else:
                    nonapp_nonspam += 1
        bayes_matrix[i][0] = smoothing(app_spam, 100)
        bayes_matrix[i][1] = smoothing(app_nonspam, 100)
        bayes_matrix[i][2] = smoothing(nonapp_spam, 100)
        bayes_matrix[i][3] = smoothing(nonapp_nonspam, 100)
    return bayes_matrix

# Hàm dự đoán mail mới
label = []
bag_Of_Words = []

for i in range(0, 200):
    if (i < 100):
        label.append(1)
    else:
        label.append(0)

train_spam = 'C:\\Users\\Admin\\PycharmProjects\\demo\\venv\\Train\\Train_spam'
train_nonspam = 'C:\\Users\\Admin\\PycharmProjects\\demo\\venv\\Train\\Train_nonspam'
set_words1 = make_Bag_Of_Words(train_spam)
set_words2 = make_Bag_Of_Words(train_nonspam)
# Tạo túi từ mà không có từ nào trùng nhau, các từ phân biệt với nhau

for word in set_words1:
    if word not in bag_Of_Words:
        bag_Of_Words.append(word)
for word in set_words2:
    if word not in bag_Of_Words:
        bag_Of_Words.append(word)

file = open('bagofwords.txt', mode='w', encoding='utf8')
for word in bag_Of_Words:
    file.write(word + " ")
file.close()

bayes_matrix =  build_matrix(train_spam, train_nonspam, bag_Of_Words)
n = len(bag_Of_Words)
file = open('bayes_matrix.txt', mode= 'a')
file.write(str(n) + "\n")
for i, v in enumerate(bayes_matrix):
    line = str(v[0]) + " " + str(v[1]) + " " + str(v[2]) + " " + str(v[3]) + "\n"
    file.write(line)
file.close()

