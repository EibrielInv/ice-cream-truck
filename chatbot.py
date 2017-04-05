import os

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

generate_dataset = False

if generate_dataset:
    possible_steps = {}
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
              batch_size=32,
              epochs=120,
              validation_data=([inputs_test, queries_test], answers_test))
    model.save_weights(model_filepath)
else:
    model.load_weights(model_filepath)

#print(inputs_test[0].reshape(1, -1).shape)
#print(queries_test[0].reshape(1, -1).shape)

print ("\n\nTest")

current_index = 1
while 1:
    print("\nStory:")
    print(list_to_string (inputs_test[current_index], vocab).replace('.', '.\n'))
    print("\nQuestion:")
    print(list_to_string (queries_test[current_index], vocab))

    prediction = model.predict([inputs_test[current_index].reshape(1, -1), queries_test[current_index].reshape(1, -1)])

    print("\nPredicted answer:")
    #print (vocab)
    #print (np_softmax(prediction))
    print (vocab[np.argmax(prediction)-1])

    current_index += 1
    input("next?")
