import requests
import gzip
import re
import random
import time
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm  # Use tqdm for progress indicators

# Download the wet.paths.gz file
wet_paths_url = 'https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-22/wet.paths.gz'
wet_paths_file = 'wet.paths.gz'

response = requests.get(wet_paths_url, stream=True)
with open(wet_paths_file, 'wb') as file:
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            file.write(chunk)

print(f"Downloaded {wet_paths_file}")

# Extract the first URL from the wet.paths.gz file
def get_first_wet_url(wet_paths_file):
    with gzip.open(wet_paths_file, 'rt', encoding='utf-8') as f:
        for line in f:
            return line.strip()

first_wet_path = get_first_wet_url(wet_paths_file)
first_wet_url = f'https://data.commoncrawl.org/{first_wet_path}'
print(f"First WET file URL: {first_wet_url}")

# Download the first WET file
wet_file_path = 'first_wet_file.warc.wet.gz'

response = requests.get(first_wet_url, stream=True)
with open(wet_file_path, 'wb') as file:
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            file.write(chunk)

print(f"Downloaded {wet_file_path}")
# Function to extract text content from a gzipped WET file
def extract_text_from_wet_file(file_path):
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        inside_text = False
        for line in f:
            if line.startswith("WARC/1.0"):
                inside_text = False
                continue
            if line.startswith("Content-Length:"):
                inside_text = True
                continue
            if inside_text:
                yield line

                # Extract text and count word frequencies
print("Extracting text from file...")
text_generator = extract_text_from_wet_file(wet_file_path)

# Function to clean text data
def clean_text(text):
    text = re.sub(r'\W+', ' ', text)  # Remove non-alphanumeric characters
    text = text.lower()  # Convert to lowercase
    return text

# Function to count word frequencies in text data
def count_word_frequencies(text_generator, target_language='ko'):
    word_counter = Counter()
    for text in text_generator:
        cleaned_text = clean_text(text)
        words = cleaned_text.split()
        word_counter.update(words)
    return word_counter

print("Counting word frequencies...")
word_frequencies = count_word_frequencies(text_generator)

# Print the number of unique words and their total frequency
total_words = sum(word_frequencies.values())
unique_words = len(word_frequencies)
print(f"Total words: {total_words}")
print(f"Unique words: {unique_words}")

# Use the word frequencies for further analysis
print(word_frequencies.most_common(10))  # Print the 10 most common words

sample_size = 15000  # Adjust sample size as needed
sampled_word_frequencies = dict(random.sample(list(word_frequencies.items()), min(sample_size, unique_words)))
sorted_word_frequencies = dict(sorted(sampled_word_frequencies.items(), key=lambda item: item[1], reverse=True))

##randomly sampled_word_frequencies

# Set the same size for deletion, adjustment, and access samples
operation_sample_size = 10 # Adjust sample size as needed

# Implementations for Array, LinkedList, and Hash Table
class Array:
    def __init__(self):
        self.array = []

    def insert(self, term):
        self.array.append(term)

    def delete(self, term):
        if term in self.array:
            self.array.remove(term)

    def access(self):
        return max(set(self.array), key=self.array.count) if self.array else None

class LinkedListNode:
    def __init__(self, term):
        self.term = term
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def insert(self, term):
        new_node = LinkedListNode(term)
        new_node.next = self.head
        self.head = new_node

    def delete(self, term):
        current = self.head
        prev = None
        while current:
            if current.term == term:
                if prev:
                    prev.next = current.next
                else:
                    self.head = current.next
                return
            prev = current
            current = current.next

    def access(self):
        frequency = {}
        current = self.head
        while current:
            if current.term in frequency:
                frequency[current.term] += 1
            else:
                frequency[current.term] = 1
            current = current.next
        return max(frequency, key=frequency.get) if frequency else None

    def print_list(self):
        current = self.head
        while current:
            print(current.term, end=" -> ")
            current = current.next
        print("None")

class TermData:
    def __init__(self, term, frequency=1):
        self.term = term
        self.frequency = frequency

    def __repr__(self):
        return f"TermData(Term={self.term}, Frequency={self.frequency})"

class FrequencyBucketNode:
    def __init__(self, term_data):
        self.term_data = term_data
        self.next = None

class FrequencyBucket:
    def __init__(self):
        self.head = None

    def insert(self, term_data):
        new_node = FrequencyBucketNode(term_data)
        new_node.next = self.head
        self.head = new_node

    def remove(self, term):
        current = self.head
        prev = None
        while current:
            if current.term_data.term == term:
                if prev:
                    prev.next = current.next
                else:
                    self.head = current.next
                return current.term_data
            prev = current
            current = current.next
        return None

class AdaptiveFrequencyHashTable:
    def __init__(self):
        self.table = {}
        self.frequency_buckets = []
        self.max_frequency = 0

    def _ensure_bucket_exists(self, frequency):
        while len(self.frequency_buckets) <= frequency:
            self.frequency_buckets.append(FrequencyBucket())

    def insert(self, term):
        if term in self.table:
            term_data = self.table[term]
            old_frequency = term_data.frequency
            self.frequency_buckets[old_frequency].remove(term)
            term_data.frequency += 1
            new_frequency = term_data.frequency
        else:
            term_data = TermData(term)
            self.table[term] = term_data
            new_frequency = term_data.frequency

        self._ensure_bucket_exists(new_frequency)
        self.frequency_buckets[new_frequency].insert(term_data)

        if new_frequency > self.max_frequency:
            self.max_frequency = new_frequency

    def delete(self, term):
        if term in self.table:
            term_data = self.table[term]
            frequency = term_data.frequency
            self.frequency_buckets[frequency].remove(term)
            del self.table[term]

    def access_high_frequency_term(self):
        if self.max_frequency == 0:
            return None
        return self.frequency_buckets[self.max_frequency].head.term_data.term

    def adjust_frequency(self, term, new_frequency):
        if term in self.table:
            term_data = self.table[term]
            old_frequency = term_data.frequency
            self.frequency_buckets[old_frequency].remove(term)
            term_data.frequency = new_frequency
            self._ensure_bucket_exists(new_frequency)
            self.frequency_buckets[new_frequency].insert(term_data)
            if new_frequency > self.max_frequency:
                self.max_frequency = new_frequency

# Performance evaluation setup
afht = AdaptiveFrequencyHashTable()
array_ds = Array()
linked_list_ds = LinkedList()
hash_table_ds = {}

print("Inserting data into data structures...")
# Insertion performance
start_time = time.time()
for word, freq in tqdm(sampled_word_frequencies.items(), desc="AFHT Insertion"):
    for _ in range(freq):
        afht.insert(word)
end_time = time.time()
afht_insertion_time = (end_time - start_time) / len(sampled_word_frequencies)

start_time = time.time()
for word, freq in tqdm(sampled_word_frequencies.items(), desc="Array Insertion"):
    for _ in range(freq):
        array_ds.insert(word)
end_time = time.time()
array_insertion_time = (end_time - start_time) / len(sampled_word_frequencies)

start_time = time.time()
for word, freq in tqdm(sampled_word_frequencies.items(), desc="LinkedList Insertion"):
    for _ in range(freq):
        linked_list_ds.insert(word)
end_time = time.time()
linked_list_insertion_time = (end_time - start_time) / len(sampled_word_frequencies)

start_time = time.time()
for word, freq in tqdm(sampled_word_frequencies.items(), desc="HashTable Insertion"):
    for _ in range(freq):
        if word in hash_table_ds:
            hash_table_ds[word] += 1
        else:
            hash_table_ds[word] = 1
end_time = time.time()
hash_table_insertion_time = (end_time - start_time) / len(sampled_word_frequencies)

print("Accessing high-frequency terms...")
# Access performance
start_time = time.time()
for _ in tqdm(range(operation_sample_size), desc="AFHT Access"):
    high_freq_term = afht.access_high_frequency_term()
end_time = time.time()
afht_access_time = (end_time - start_time) / operation_sample_size

start_time = time.time()
for _ in tqdm(range(operation_sample_size), desc="Array Access"):
    high_freq_term = array_ds.access()
end_time = time.time()
array_access_time = (end_time - start_time) / operation_sample_size

start_time = time.time()
for _ in tqdm(range(operation_sample_size), desc="LinkedList Access"):
    high_freq_term = linked_list_ds.access()
end_time = time.time()
linked_list_access_time = (end_time - start_time) / operation_sample_size

start_time = time.time()
for _ in tqdm(range(operation_sample_size), desc="HashTable Access"):
    high_freq_term = max(hash_table_ds, key=hash_table_ds.get) if hash_table_ds else None
end_time = time.time()
hash_table_access_time = (end_time - start_time) / operation_sample_size

print("Adjusting frequencies...")
# Adjustment performance
words_to_adjust = random.sample(list(sampled_word_frequencies.keys()), min(operation_sample_size, len(sampled_word_frequencies)))

start_time = time.time()
for word in tqdm(words_to_adjust, desc="AFHT Adjustment"):
    new_frequency = random.randint(1, 100)
    afht.adjust_frequency(word, new_frequency)
end_time = time.time()
afht_adjustment_time = (end_time - start_time) / len(words_to_adjust)

start_time = time.time()
for word in tqdm(words_to_adjust, desc="Array Adjustment"):
    array_ds.insert(word)
    for _ in range(random.randint(1, 100)):
        array_ds.insert(word)
end_time = time.time()
array_adjustment_time = (end_time - start_time) / len(words_to_adjust)

start_time = time.time()
for word in tqdm(words_to_adjust, desc="LinkedList Adjustment"):
    linked_list_ds.insert(word)
    for _ in range(random.randint(1, 100)):
        linked_list_ds.insert(word)
end_time = time.time()
linked_list_adjustment_time = (end_time - start_time) / len(words_to_adjust)

start_time = time.time()
for word in tqdm(words_to_adjust, desc="HashTable Adjustment"):
    if word in hash_table_ds:
        hash_table_ds[word] += random.randint(1, 100)
    else:
        hash_table_ds[word] = random.randint(1, 100)
end_time = time.time()
hash_table_adjustment_time = (end_time - start_time) / len(words_to_adjust)

print("Deleting data from data structures...")
# Deletion performance
words_to_delete = random.sample(list(sampled_word_frequencies.keys()), min(operation_sample_size, len(sampled_word_frequencies)))

start_time = time.time()
for word in tqdm(words_to_delete, desc="AFHT Deletion"):
    afht.delete(word)
end_time = time.time()
afht_deletion_time = (end_time - start_time) / len(words_to_delete)

start_time = time.time()
for word in tqdm(words_to_delete, desc="Array Deletion"):
    array_ds.delete(word)
end_time = time.time()
array_deletion_time = (end_time - start_time) / len(words_to_delete)

start_time = time.time()
for word in tqdm(words_to_delete, desc="LinkedList Deletion"):
    linked_list_ds.delete(word)
end_time = time.time()
linked_list_deletion_time = (end_time - start_time) / len(words_to_delete)

start_time = time.time()
for word in tqdm(words_to_delete, desc="HashTable Deletion"):
    if word in hash_table_ds:
        del hash_table_ds[word]
end_time = time.time()
hash_table_deletion_time = (end_time - start_time) / len(words_to_delete)

# Print results
print(f"AFHT Insertion time per term: {afht_insertion_time:.6f} seconds")
print(f"AFHT Deletion time per term: {afht_deletion_time:.6f} seconds")
print(f"AFHT Access time per term: {afht_access_time:.6f} seconds")
print(f"AFHT Adjustment time per term: {afht_adjustment_time:.6f} seconds")

print(f"Array Insertion time per term: {array_insertion_time:.6f} seconds")
print(f"Array Deletion time per term: {array_deletion_time:.6f} seconds")
print(f"Array Access time per term: {array_access_time:.6f} seconds")
print(f"Array Adjustment time per term: {array_adjustment_time:.6f} seconds")

print(f"Linked List Insertion time per term: {linked_list_insertion_time:.6f} seconds")
print(f"Linked List Deletion time per term: {linked_list_deletion_time:.6f} seconds")
print(f"Linked List Access time per term: {linked_list_access_time:.6f} seconds")
print(f"Linked List Adjustment time per term: {linked_list_adjustment_time:.6f} seconds")

print(f"Hash Table Insertion time per term: {hash_table_insertion_time:.6f} seconds")
print(f"Hash Table Deletion time per term: {hash_table_deletion_time:.6f} seconds")
print(f"Hash Table Access time per term: {hash_table_access_time:.6f} seconds")
print(f"Hash Table Adjustment time per term: {hash_table_adjustment_time:.6f} seconds")

# Performance comparison table
performance_data = {
    'Operation': ['Insertion', 'Deletion', 'Access', 'Adjustment'],
    'Array': [array_insertion_time, array_deletion_time, array_access_time, array_adjustment_time],
    'Linked List': [linked_list_insertion_time, linked_list_deletion_time, linked_list_access_time, linked_list_adjustment_time],
    'Hash Table': [hash_table_insertion_time, hash_table_deletion_time, hash_table_access_time, hash_table_adjustment_time],
    'AFHT': [afht_insertion_time, afht_deletion_time, afht_access_time, afht_adjustment_time]
}

# Create performance comparison graph
operations = ['Insertion', 'Deletion', 'Access', 'Adjustment', 'Access+Adjustment']
array_times = [array_insertion_time, array_deletion_time, array_access_time, array_adjustment_time, array_access_time+array_adjustment_time]
linked_list_times = [linked_list_insertion_time, linked_list_deletion_time, linked_list_access_time, linked_list_adjustment_time, linked_list_access_time+linked_list_adjustment_time]
hash_table_times = [hash_table_insertion_time, hash_table_deletion_time, hash_table_access_time, hash_table_adjustment_time,hash_table_access_time+hash_table_adjustment_time]
afht_times = [afht_insertion_time, afht_deletion_time, afht_access_time, afht_adjustment_time, afht_access_time+afht_adjustment_time]

x = range(len(operations))

plt.figure(figsize=(10, 6))
plt.plot(x, array_times, label='Array', marker='o')
plt.plot(x, linked_list_times, label='Linked List', marker='o')
plt.plot(x, hash_table_times, label='Hash Table', marker='o')
plt.plot(x, afht_times, label='AFHT', marker='o')


plt.xticks(x, operations)
plt.xlabel('Operations')
plt.ylabel('Time (log scale, seconds)')
plt.yscale('log')  # Set the y-axis to logarithmic scale
plt.title('Time Complexity Comparison of Data Structures (Log Scale)')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()

##sorted_word_frequencies

# Set the same size for deletion, adjustment, and access samples
operation_sample_size = 3 # Adjust sample size as needed

# Implementations for Array, LinkedList, and Hash Table
class Array:
    def __init__(self):
        self.array = []

    def insert(self, term):
        self.array.append(term)

    def delete(self, term):
        if term in self.array:
            self.array.remove(term)

    def access(self):
        return max(set(self.array), key=self.array.count) if self.array else None

class LinkedListNode:
    def __init__(self, term):
        self.term = term
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def insert(self, term):
        new_node = LinkedListNode(term)
        new_node.next = self.head
        self.head = new_node

    def delete(self, term):
        current = self.head
        prev = None
        while current:
            if current.term == term:
                if prev:
                    prev.next = current.next
                else:
                    self.head = current.next
                return
            prev = current
            current = current.next

    def access(self):
        frequency = {}
        current = self.head
        while current:
            if current.term in frequency:
                frequency[current.term] += 1
            else:
                frequency[current.term] = 1
            current = current.next
        return max(frequency, key=frequency.get) if frequency else None

class TermData:
    def __init__(self, term, frequency=1):
        self.term = term
        self.frequency = frequency

    def __repr__(self):
        return f"TermData(Term={self.term}, Frequency={self.frequency})"

class FrequencyBucketNode:
    def __init__(self, term_data):
        self.term_data = term_data
        self.next = None

class FrequencyBucket:
    def __init__(self):
        self.head = None

    def insert(self, term_data):
        new_node = FrequencyBucketNode(term_data)
        new_node.next = self.head
        self.head = new_node

    def remove(self, term):
        current = self.head
        prev = None
        while current:
            if current.term_data.term == term:
                if prev:
                    prev.next = current.next
                else:
                    self.head = current.next
                return current.term_data
            prev = current
            current = current.next
        return None

class AdaptiveFrequencyHashTable:
    def __init__(self):
        self.table = {}
        self.frequency_buckets = []
        self.max_frequency = 0

    def _ensure_bucket_exists(self, frequency):
        while len(self.frequency_buckets) <= frequency:
            self.frequency_buckets.append(FrequencyBucket())

    def insert(self, term):
        if term in self.table:
            term_data = self.table[term]
            old_frequency = term_data.frequency
            self.frequency_buckets[old_frequency].remove(term)
            term_data.frequency += 1
            new_frequency = term_data.frequency
        else:
            term_data = TermData(term)
            self.table[term] = term_data
            new_frequency = term_data.frequency

        self._ensure_bucket_exists(new_frequency)
        self.frequency_buckets[new_frequency].insert(term_data)

        if new_frequency > self.max_frequency:
            self.max_frequency = new_frequency

    def delete(self, term):
        if term in self.table:
            term_data = self.table[term]
            frequency = term_data.frequency
            self.frequency_buckets[frequency].remove(term)
            del self.table[term]

    def access_high_frequency_term(self):
        if self.max_frequency == 0:
            return None
        return self.frequency_buckets[self.max_frequency].head.term_data.term

    def adjust_frequency(self, term, new_frequency):
        if term in self.table:
            term_data = self.table[term]
            old_frequency = term_data.frequency
            self.frequency_buckets[old_frequency].remove(term)
            term_data.frequency = new_frequency
            self._ensure_bucket_exists(new_frequency)
            self.frequency_buckets[new_frequency].insert(term_data)
            if new_frequency > self.max_frequency:
                self.max_frequency = new_frequency

# Performance evaluation setup
afht = AdaptiveFrequencyHashTable()
array_ds = Array()
linked_list_ds = LinkedList()
hash_table_ds = {}

print("Inserting data into data structures...")
# Insertion performance
start_time = time.time()
for word, freq in tqdm(sorted_word_frequencies.items(), desc="AFHT Insertion"):
    for _ in range(freq):
        afht.insert(word)
end_time = time.time()
afht_insertion_time = (end_time - start_time) / len(sorted_word_frequencies)

start_time = time.time()
for word, freq in tqdm(sorted_word_frequencies.items(), desc="Array Insertion"):
    for _ in range(freq):
        array_ds.insert(word)
end_time = time.time()
array_insertion_time = (end_time - start_time) / len(sorted_word_frequencies)

start_time = time.time()
for word, freq in tqdm(sorted_word_frequencies.items(), desc="LinkedList Insertion"):
    for _ in range(freq):
        linked_list_ds.insert(word)
end_time = time.time()
linked_list_insertion_time = (end_time - start_time) / len(sorted_word_frequencies)

start_time = time.time()
for word, freq in tqdm(sorted_word_frequencies.items(), desc="HashTable Insertion"):
    for _ in range(freq):
        if word in hash_table_ds:
            hash_table_ds[word] += 1
        else:
            hash_table_ds[word] = 1
end_time = time.time()
hash_table_insertion_time = (end_time - start_time) / len(sorted_word_frequencies)

print("Accessing high-frequency terms...")
# Access performance
start_time = time.time()
for _ in tqdm(range(operation_sample_size), desc="AFHT Access"):
    high_freq_term = afht.access_high_frequency_term()
end_time = time.time()
afht_access_time = (end_time - start_time) / operation_sample_size

start_time = time.time()
for _ in tqdm(range(operation_sample_size), desc="Array Access"):
    high_freq_term = array_ds.access()
end_time = time.time()
array_access_time = (end_time - start_time) / operation_sample_size

start_time = time.time()
for _ in tqdm(range(operation_sample_size), desc="LinkedList Access"):
    high_freq_term = linked_list_ds.access()
end_time = time.time()
linked_list_access_time = (end_time - start_time) / operation_sample_size

start_time = time.time()
for _ in tqdm(range(operation_sample_size), desc="HashTable Access"):
    high_freq_term = max(hash_table_ds, key=hash_table_ds.get) if hash_table_ds else None
end_time = time.time()
hash_table_access_time = (end_time - start_time) / operation_sample_size

print("Adjusting frequencies...")
# Adjustment performance
words_to_adjust = random.sample(list(sorted_word_frequencies.keys()), min(operation_sample_size, len(sorted_word_frequencies)))

start_time = time.time()
for word in tqdm(words_to_adjust, desc="AFHT Adjustment"):
    new_frequency = random.randint(1, 100)
    afht.adjust_frequency(word, new_frequency)
end_time = time.time()
afht_adjustment_time = (end_time - start_time) / len(words_to_adjust)

start_time = time.time()
for word in tqdm(words_to_adjust, desc="Array Adjustment"):
    array_ds.insert(word)
    for _ in range(random.randint(1, 100)):
        array_ds.insert(word)
end_time = time.time()
array_adjustment_time = (end_time - start_time) / len(words_to_adjust)

start_time = time.time()
for word in tqdm(words_to_adjust, desc="LinkedList Adjustment"):
    linked_list_ds.insert(word)
    for _ in range(random.randint(1, 100)):
        linked_list_ds.insert(word)
end_time = time.time()
linked_list_adjustment_time = (end_time - start_time) / len(words_to_adjust)

start_time = time.time()
for word in tqdm(words_to_adjust, desc="HashTable Adjustment"):
    if word in hash_table_ds:
        hash_table_ds[word] += random.randint(1, 100)
    else:
        hash_table_ds[word] = random.randint(1, 100)
end_time = time.time()
hash_table_adjustment_time = (end_time - start_time) / len(words_to_adjust)

print("Deleting data from data structures...")
# Deletion performance
words_to_delete = random.sample(list(sorted_word_frequencies.keys()), min(operation_sample_size, len(sorted_word_frequencies)))

start_time = time.time()
for word in tqdm(words_to_delete, desc="AFHT Deletion"):
    afht.delete(word)
end_time = time.time()
afht_deletion_time = (end_time - start_time) / len(words_to_delete)

start_time = time.time()
for word in tqdm(words_to_delete, desc="Array Deletion"):
    array_ds.delete(word)
end_time = time.time()
array_deletion_time = (end_time - start_time) / len(words_to_delete)

start_time = time.time()
for word in tqdm(words_to_delete, desc="LinkedList Deletion"):
    linked_list_ds.delete(word)
end_time = time.time()
linked_list_deletion_time = (end_time - start_time) / len(words_to_delete)

start_time = time.time()
for word in tqdm(words_to_delete, desc="HashTable Deletion"):
    if word in hash_table_ds:
        del hash_table_ds[word]
end_time = time.time()
hash_table_deletion_time = (end_time - start_time) / len(words_to_delete)

# Print results
print(f"AFHT Insertion time per term: {afht_insertion_time:.6f} seconds")
print(f"AFHT Deletion time per term: {afht_deletion_time:.6f} seconds")
print(f"AFHT Access time per term: {afht_access_time:.6f} seconds")
print(f"AFHT Adjustment time per term: {afht_adjustment_time:.6f} seconds")

print(f"Array Insertion time per term: {array_insertion_time:.6f} seconds")
print(f"Array Deletion time per term: {array_deletion_time:.6f} seconds")
print(f"Array Access time per term: {array_access_time:.6f} seconds")
print(f"Array Adjustment time per term: {array_adjustment_time:.6f} seconds")

print(f"Linked List Insertion time per term: {linked_list_insertion_time:.6f} seconds")
print(f"Linked List Deletion time per term: {linked_list_deletion_time:.6f} seconds")
print(f"Linked List Access time per term: {linked_list_access_time:.6f} seconds")
print(f"Linked List Adjustment time per term: {linked_list_adjustment_time:.6f} seconds")

print(f"Hash Table Insertion time per term: {hash_table_insertion_time:.6f} seconds")
print(f"Hash Table Deletion time per term: {hash_table_deletion_time:.6f} seconds")
print(f"Hash Table Access time per term: {hash_table_access_time:.6f} seconds")
print(f"Hash Table Adjustment time per term: {hash_table_adjustment_time:.6f} seconds")

# Performance comparison table
performance_data = {
    'Operation': ['Insertion', 'Deletion', 'Access', 'Adjustment'],
    'Array': [array_insertion_time, array_deletion_time, array_access_time, array_adjustment_time],
    'Linked List': [linked_list_insertion_time, linked_list_deletion_time, linked_list_access_time, linked_list_adjustment_time],
    'Hash Table': [hash_table_insertion_time, hash_table_deletion_time, hash_table_access_time, hash_table_adjustment_time],
    'AFHT': [afht_insertion_time, afht_deletion_time, afht_access_time, afht_adjustment_time]
}

# Create performance comparison graph
operations = ['Insertion', 'Deletion', 'Access', 'Adjustment', 'Access+Adjustment']
array_times = [array_insertion_time, array_deletion_time, array_access_time, array_adjustment_time, array_access_time+array_adjustment_time]
linked_list_times = [linked_list_insertion_time, linked_list_deletion_time, linked_list_access_time, linked_list_adjustment_time, linked_list_access_time+linked_list_adjustment_time]
hash_table_times = [hash_table_insertion_time, hash_table_deletion_time, hash_table_access_time, hash_table_adjustment_time,hash_table_access_time+hash_table_adjustment_time]
afht_times = [afht_insertion_time, afht_deletion_time, afht_access_time, afht_adjustment_time, afht_access_time+afht_adjustment_time]

x = range(len(operations))

plt.figure(figsize=(10, 6))
plt.plot(x, array_times, label='Array', marker='o')
plt.plot(x, linked_list_times, label='Linked List', marker='o')
plt.plot(x, hash_table_times, label='Hash Table', marker='o')
plt.plot(x, afht_times, label='AFHT', marker='o')


plt.xticks(x, operations)
plt.xlabel('Operations')
plt.ylabel('Time (log scale, seconds)')
plt.yscale('log')  # Set the y-axis to logarithmic scale
plt.title('Time Complexity Comparison of Data Structures (Log Scale)')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()
