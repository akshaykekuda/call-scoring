import sys
import json
import ast
import nltk
from nltk.tokenize import sent_tokenize 
import unidecode
import contractions
import matplotlib.pyplot as plt

file_in = open('yelp_academic_dataset_review.json', 'r', encoding="utf8")


stars_before = []
stars_after = []
sents_before = []
sents_after = []


def generate_file(filename, num_entries):

    print("Start writing to " + filename)
    file_out = open(filename, 'w')
    i = 0
    while i < num_entries:
        line = file_in.readline()
        try:
            json_line = json.loads(line)

            text = handle_punctuation(json_line['text'])

            num_sents = len(sent_tokenize(text))

            stars_before.append(json_line['stars'])
            sents_before.append(num_sents)


            if num_sents >= 3 and num_sents <= 20:
                new_data = {}
                valid = False
                if i < num_entries / 2 and json_line['stars'] >= 3:
                    new_data['category'] = 1
                    valid = True
                elif i >= num_entries / 2 and json_line['stars'] < 3:
                    new_data['category'] = 0
                    valid = True

                if valid:
                    text = clean_text(text)
                    num_sents = len(sent_tokenize(text))
                    if num_sents >= 3 and num_sents <= 20: #In case cleaning text altered length
                        stars_after.append(json_line['stars'])
                        sents_after.append(num_sents)
                        new_data['text'] = clean_text(text)
                        json.dump(new_data, file_out)
                        file_out.write('\n')
                        i += 1

        except Exception as e:
            print(e)

    print("Done writing to " + filename)
    file_out.close()

def clean_text(text):

    #Get rid of accents
    text = unidecode.unidecode(text)

    #Expand contractions
    text = contractions.fix(text)

    #Remove special characters
    text = ''.join(c if c.isalnum() or c.isspace() or c == '.' else ' ' for c in text)

    #Make everything lowercase
    text = text.lower()

    #Make all whitespace into space
    text = ' '.join(text.split())
    
    return text

def handle_punctuation(text):
    #Convert ? and ! to .
    text = text.replace('!', '.')
    text = text.replace('?', '.')

    #Get rid of duplicate punctuation
    last_period = False
    temp_text = ''
    for c in text:
        if c != '.':
            last_period = False
            temp_text += c
        elif not last_period and not c.isspace():
            last_period = True
            temp_text += c + ' '
    text = temp_text

    return text


generate_file('dataset_train.json', 50000)
generate_file('dataset_dev.json', 10000)
generate_file('dataset_test.json', 8000)

file_in.close()

stars_before_map = dict()
stars_after_map = dict()
sents_before_map = dict()
sents_after_map = dict()

for i in range(len(stars_before)):
    if stars_before[i] not in stars_before_map:
        stars_before_map[stars_before[i]] = 1
    else:
        stars_before_map[stars_before[i]] += 1
    if sents_before[i] not in sents_before_map:
        sents_before_map[sents_before[i]] = 1
    else:
        sents_before_map[sents_before[i]] += 1

for i in range(len(stars_after)):
    if stars_after[i] not in stars_after_map:
        stars_after_map[stars_after[i]] = 1
    else:
        stars_after_map[stars_after[i]] += 1
    if sents_after[i] not in sents_after_map:
        sents_after_map[sents_after[i]] = 1
    else:
        sents_after_map[sents_after[i]] += 1

print("Stars Before:")
print(stars_before_map)
print("\nStars After:")
print(stars_after_map)
print("\nNumber of Sentences Before:")
print(sents_before_map)
print("\nNumber of Sentences After:")
print(sents_after_map)

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
# plt.suptitle("Yelp Dataset Statistics", fontsize=14)

stars_before_values = [stars_before_map[1], stars_before_map[2], stars_before_map[3], stars_before_map[4], stars_before_map[5]]
stars_after_values = [stars_after_map[1], stars_after_map[2], stars_after_map[3], stars_after_map[4], stars_after_map[5]]


axs[0, 0].bar([1, 2, 3, 4, 5], stars_before_values)
axs[0, 0].set(xlabel='Stars', ylabel='Frequency')
axs[0, 0].set_title('Stars Before Preprocessing')

axs[0, 1].bar([1, 2, 3, 4, 5], stars_after_values)
axs[0, 1].set(xlabel='Stars', ylabel='Frequency')
axs[0, 1].set_title('Stars After Preprocessing')

axs[1, 0].hist(x=sents_before)
axs[1, 0].set(xlabel='Number of Sentences', ylabel='Frequency')
axs[1, 0].set_title('Number of Sentences Before Preprocessing')

axs[1, 1].hist(x=sents_after)
axs[1, 1].set(xlabel='Number of Sentences', ylabel='Frequency')
axs[1, 1].set_title('Number of Sentences After Preprocessing')

plt.tight_layout()
plt.show()