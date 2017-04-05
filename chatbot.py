import os
import random as rd

from keras.layers import add
from keras.layers import dot
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Dropout
from keras.layers import Permute
from keras.layers import Activation
from keras.layers import concatenate

from keras.models import Model
from keras.models import Sequential
from keras.layers.embeddings import Embedding

from helpers import *

from telegram import telegram

from config import Config

generate_dataset = True

if generate_dataset:
    add_commands = [
        'add',
        'i would like',
        'i want'
    ]
    remove_commands = [
        'remove',
        'i dont want'
    ]
    change_commands = [
        'change:for'
    ]
    flavors = [
        'chocolate',
        'lemon',
        'cherry',
        'coffee'
    ]
    generated_dataset = []
    stories_count = 40000
    for n in range(stories_count):
        is_flavor = [False, False, False, False]
        sentences = []
        for n in range(rd.randint(1, 6)):
            #if n > 0:
            #    random_action = rd.randint(0, 2)
            #else:
            #    random_action = 0 # always add first
            random_action = rd.randint(0, 2)
            random_flavor = rd.randint(0, 3)
            random_flavor_b = rd.randint(0, 3)
            if random_action==0: #add
                is_flavor[random_flavor] = True
                text = "{} {} .".format(rd.choice(add_commands), flavors[random_flavor])
            elif random_action==1: #remove
                is_flavor[random_flavor] = False
                text = "{} {} .".format(rd.choice(remove_commands), flavors[random_flavor])
            elif random_action==2: #change
                is_flavor[random_flavor] = False
                is_flavor[random_flavor_b] = True
                command_text = rd.choice(change_commands)
                command_text = command_text.split(':')
                text = "{} {} {} {} .".format(command_text[0], flavors[random_flavor], command_text[1], flavors[random_flavor_b])
            sentences.append(text)
        sentences = " ".join(sentences)
        random_flavor = rd.randint(0, 3)
        question = "is {} in the order ?".format(flavors[random_flavor])
        answer = "yes" if is_flavor[random_flavor] else "no"
        generated_dataset.append( (sentences.split(" "), question.split(" "), answer) )

    split_idx = int(stories_count/2)
    train_stories = generated_dataset[:split_idx]
    test_stories = generated_dataset[split_idx:]
else:
    challenges = {
        # QA1 with 10,000 samples
        'single_supporting_fact_10k': 'data/tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt',
        # QA2 with 10,000 samples
        'two_supporting_facts_10k': 'data/tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt',
    }
    challenge_type = 'single_supporting_fact_10k'
    challenge = challenges[challenge_type]

    print('Extracting stories for the challenge:', challenge_type)

    train_stories = get_stories(open(challenge.format('train'), 'r'))
    test_stories = get_stories(open(challenge.format('test'), 'r'))

vocab = set()
for story, q, answer in train_stories + test_stories:
    vocab |= set(story + q + [answer])
vocab = sorted(vocab)

# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1
story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))

print('-')
print('Vocab size:', vocab_size, 'unique words')
print('Story max length:', story_maxlen, 'words')
print('Query max length:', query_maxlen, 'words')
print('Number of training stories:', len(train_stories))
print('Number of test stories:', len(test_stories))
print('-')
print('Here\'s what a "story" tuple looks like (input, query, answer):')
print(train_stories[0])
print('-')
print('Vectorizing the word sequences...')

word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
inputs_train, queries_train, answers_train = vectorize_stories(train_stories,
                                                               word_idx,
                                                               story_maxlen,
                                                               query_maxlen)
inputs_test, queries_test, answers_test = vectorize_stories(test_stories,
                                                            word_idx,
                                                            story_maxlen,
                                                            query_maxlen)

print('-')
print('inputs: integer tensor of shape (samples, max_length)')
print('inputs_train shape:', inputs_train.shape)
print('inputs_test shape:', inputs_test.shape)
print('-')
print('queries: integer tensor of shape (samples, max_length)')
print('queries_train shape:', queries_train.shape)
print('queries_test shape:', queries_test.shape)
print('-')
print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
print('answers_train shape:', answers_train.shape)
print('answers_test shape:', answers_test.shape)
print('-')
print('Compiling...')


#story_maxlen = 1

embed_dim=64
keep_prob = 0.3

# placeholders
input_sequence = Input((story_maxlen,))
question = Input((query_maxlen,))


# encoders
# embed the input sequence into a sequence of vectors
input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim=vocab_size,
                              output_dim=embed_dim))
input_encoder_m.add(Dropout(keep_prob))
# output: (samples, story_maxlen, embedding_dim)

# embed the input into a sequence of vectors of size query_maxlen
input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim=vocab_size,
                              output_dim=query_maxlen))
input_encoder_c.add(Dropout(keep_prob))
# output: (samples, story_maxlen, query_maxlen)

# embed the question into a sequence of vectors
question_encoder = Sequential()
question_encoder.add(Embedding(input_dim=vocab_size,
                               output_dim=embed_dim,
                               input_length=query_maxlen))
question_encoder.add(Dropout(keep_prob))
# output: (samples, query_maxlen, embedding_dim)


# encode input sequence and questions (which are indices)
# to sequences of dense vectors
input_encoded_m = input_encoder_m(input_sequence)
input_encoded_c = input_encoder_c(input_sequence)
question_encoded = question_encoder(question)


# compute a 'match' between the first input vector sequence
# and the question vector sequence
# shape: `(samples, story_maxlen, query_maxlen)`
match = dot([input_encoded_m, question_encoded], axes=(2, 2))
match = Activation('softmax')(match)

# add the match matrix with the second input vector sequence
response = add([match, input_encoded_c])  # (samples, story_maxlen, query_maxlen)
response = Permute((2, 1))(response)  # (samples, query_maxlen, story_maxlen)

# concatenate the match matrix with the question vector sequence
answer = concatenate([response, question_encoded])

# the original paper uses a matrix multiplication for this reduction step.
# we choose to use a RNN instead.
answer = LSTM(32)(answer)  # (samples, 32)

# one regularization layer -- more would probably be needed.
answer = Dropout(keep_prob)(answer)
answer = Dense(vocab_size)(answer)  # (samples, vocab_size)
# we output a probability distribution over the vocabulary
answer = Activation('softmax')(answer)


# build the final model
model = Model([input_sequence, question], answer)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])

model_filepath = 'model.hdf5'
if not os.path.isfile(model_filepath):
    # train
    model.fit([inputs_train, queries_train], answers_train,
              batch_size=32*8,
              epochs=120,
              validation_data=([inputs_test, queries_test], answers_test))
    model.save_weights(model_filepath)
else:
    model.load_weights(model_filepath)

#print(inputs_test[0].reshape(1, -1).shape)
#print(queries_test[0].reshape(1, -1).shape)

print ("\n\nTest")

if 0:
    current_index = 1
    while 1:
        print("\nStory:")
        print(list_to_string (inputs_test[current_index], vocab).replace('.', '.\n'))
        print("\nQuestion:")
        print(list_to_string (queries_test[current_index], vocab))

        print (inputs_test[current_index])
        print (queries_test[current_index])

        prediction = model.predict([inputs_test[current_index].reshape(1, -1), queries_test[current_index].reshape(1, -1)])

        print("\nPredicted answer:")
        #print (vocab)
        #print (np_softmax(prediction))
        print (vocab[np.argmax(prediction)-1])

        current_index += 1
        input("next?")


def order_from_story(story):
    story_int = [0 for n in range(30-len(story))]
    story_int = story_int + [word_idx[word] for word in story]

    order = []
    for f in flavors:
        query = ["is", f, "in", "the", "order", "?"]
        query_int = [word_idx[word] for word in query]
        prediction = model.predict([np.array(story_int).reshape(1, -1), np.array(query_int).reshape(1, -1)])
        if vocab[np.argmax(prediction)-1]=="yes":
            order.append(f)
    order = ", ".join(order)
    return "Your order: *{}* 🍦\n\n(restarting order)\n\n".format(order)

def send_to_telegram(chat_id, answer):
    msg = {
            'chat_id': chat_id,
            'parse_mode': 'Markdown',
            'text': answer,
        }
    r = telegram_conection.send_to_bot('sendMessage', data = msg)

def known_words(sentence):
    for word in sentence:
        if not word in word_idx or word in ["order", "yes", "no", "is", "in", "the"]:
            return False
    return True

welcome_text = """

*Welcome to the End-to-End Ice Cream Truck, please place your order.*
I understand the following commands:
*add flavor* / *i would like flavor* / *i want flavor*
To select a new flavor
*remove flavor* / *i dont want flavor*
To remove a selected flavor
*change flavor for flavor*
To change one flavor to another
*done* - To print your current order
*quit* - To exit

Today flavors: _chocolate_ - _lemon_ - _cherry_ - _coffee_
"""

ui="cli"
if ui=="cli":
    input_text = ""
    print (welcome_text)
    while 1:
        story = []
        while 1:
            input_text = input(">")
            if input_text in ["done", "quit", "order"]:
                break
            sentence = input_text.split(" ")
            if sentence[-1] != ".":
                sentence.append(".")
            if not known_words(sentence):
                print ("Unknown command")
                continue
            story = story + sentence
        if input_text == "quit":
            break
        print ("\n")
        print (order_from_story(story))

elif ui=="telegram":
    telegram_conection = telegram("eibriel_icecream_bot", Config.telegram_token, "8979")
    chat_history = {}
    while 1:
        telegram_conection.open_session()
        r = telegram_conection.get_update()
        if not r:
            continue
        r_json = r.json()
        telegram_conection.close_session()
        for result in r_json["result"]:
            answer = ""
            if not ("message" in result and "text" in result["message"]):
                continue
            chat_id = result["message"]["chat"]["id"]
            sentence = result["message"]["text"].lower()

            if sentence == "/restart":
                chat_history[chat_id] = []
                send_to_telegram(chat_id, "Order restarted")
                continue

            if sentence in ["done", "quit", "order"]:
                answer = order_from_story(chat_history[chat_id])
                send_to_telegram(chat_id, answer)
                chat_history[chat_id] = []
                continue

            sentence = sentence.split(" ")
            if sentence[-1] != ".":
                sentence.append(".")

            if not known_words(sentence):
                send_to_telegram(chat_id, welcome_text)
                continue

            if not chat_id in chat_history:
                chat_history[chat_id] = []
            chat_history[chat_id] += sentence
            send_to_telegram(chat_id, "Ok!")
