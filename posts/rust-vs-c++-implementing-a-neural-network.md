<!--
.. title: Rust vs C++ - Implementing a Neural Network
.. slug: rust-vs-c++-implementing-a-neural-network
.. date: 2019-07-06 19:18:48 UTC+02:00
.. tags: 
.. previewimage: /files/cpprust.png
.. category: 
.. link: 
.. description: 
.. pretty_url: False
.. type: text
-->

I first learned Rust back in 2014, before it was stable. Rust is definitely a very interesting language so I have decided to revisit it by programming a simple neural network. For comparison, I also implemented the network in C++, the language I'm looking to replace.
<!-- TEASER_END -->

<img src="/files/cpprust.svg" width="60%" style="display:block; padding:1em 0; margin: auto;"/>

I like learning programming languages that use constructs or paradigms that are fundamentally different from what I have seen before. Rust was one such language when I first learned programming in it back in 2014. Rust is a multi-paradigm language. It features an interesting, and in my opinion superior, take on object oriented programming with its trait system and the syntactic separation of data and methods, as well as functional language features, which are starting to become more popular in other languages too. One of Rust's defining features is its memory management model, which uses a very sophisticated *borrow checker* that manages lifetimes and ownership. Memory safety is central to Rust and it achieves this without using garbage collection.

What attracted me to Rust the most was not its promise of memory safety, although the borrow checker was certainly interesting, but its functional programming features, such as [enums](https://doc.rust-lang.org/book/ch06-00-enums.html) and [pattern matching](https://doc.rust-lang.org/book/ch18-00-patterns.html), along with the amazing package manager, cargo, that comes with the default Rust installation, with the promise of no compromise in performance.

Although Rust stable was released in May 15 2015, I didn't program in Rust anymore because I was mainly interested in high performance, low-level programming and Rust was still not very close to the performance I was getting in C++ and it didn't support SIMD in stable until April 2018. There were also other small annoyances, like [explicitly having to cast in cases where there is no harm in implicitly casting](https://github.com/rust-lang/rust/issues/18878), but most of them just take some time getting used to. On the other hand, although I have been programming in C++ for quite some time now, it never felt right -- I actually enjoy programming in plain C more. Its object-oriented programming model has been the source of a lot of controversy and confusion, and there is way to much flexibility in the language making it easy to get lost in implementation details. Recent revisions of the language by the standard committee have added new, modern features, but have at the same time increased the complexity of the language with features such as rvalue references and template explosion (I suggested you watch [this](https://www.youtube.com/watch?v=PNRju6_yn3o) to get an idea of the madness), and increased the barrier to entry for beginners.

Given the current state of C++ is probably not going to improve, I have an itch that remains to be scratched. After being out of touch with Rust for over 5 years, I have basically forgotten most of the syntax, so I had a quick refresher using the fantastic [Rust book](https://doc.rust-lang.org/book/).

For the implementation of the neural network I followed the first two chapters of [this](http://neuralnetworksanddeeplearning.com/index.html) book. I will not cover any theory behind neural networks, so if you'd like to learn more please read it. I started with the Rust implementation so that I could learn more by having to think about how to solve problems. The niceties of Rust come into play from the very start of the project. Cargo allows quickly starting up a new project using `cargo new` and handles creating a directory tree, creating a [manifest file](https://doc.rust-lang.org/cargo/reference/manifest.html) and a main file, and initializing a new git repository. When learning a new language, there is a lot of head scratching involved in the beginning, and that's especially so with Rust where you have to fight the borrow checker until you get used to the fact that you cannot just pass around variables like in other languages, but you also have to declare your intentions. Another minor annoyance is that because a lot of the functionality on types is built using traits, it's difficult for a newcomer to browse through the [API reference](https://doc.rust-lang.org/std/), and easily find all the functionality supported by a certain type; you have to look at the traits that are implemented and you need to know what each trait provides.

# Parsing the MNIST database

The first thing I programmed was a parser for the binary format of the [MNIST database](http://yann.lecun.com/exdb/mnist/). The database is conveniently separated into a test and a training dataset, and each set contains an images file and a labels file. The file formats are pretty straightforward; we need to just read some bytes in, and reverse the order of bytes since the file are stored in big endian. Here is my implementation of the function for reading the labels file,
``` Rust
fn read_labels<P: AsRef<Path>>(path: P) -> std::io::Result<Vec<u8>>
{
    let mut fs_labels = std::fs::File::open(path)?;

    let mut magic_number: u32 = 0;
    let mut num_items:    u32 = 0;
    unsafe
    {
        let mut u32_bytes = std::slice::from_raw_parts_mut(
            &mut magic_number as *mut u32 as *mut u8, 4
        );
        fs_labels.read_exact(&mut u32_bytes)?;
        u32_bytes.reverse();

        assert!(magic_number == 0x801);

        let mut u32_bytes = std::slice::from_raw_parts_mut(
            &mut num_items as *mut u32 as *mut u8, 4
        );
        fs_labels.read_exact(&mut u32_bytes)?;
        u32_bytes.reverse();
    }

    let mut labels: Vec<u8> = Vec::new();
    fs_labels.read_to_end(&mut labels)?;

    assert!(labels.len() == num_items as usize);

    Ok(labels)
}
```
Let's start with the function declaration,
``` Rust
fn read_labels<P: AsRef<Path>>(path: P) -> std::io::Result<Vec<u8>>
```
Here we already see the use of a generic, let me explain what this does. The generic is [bounded](https://doc.rust-lang.org/rust-by-example/generics/bounds.html) with the trait `AsRef<Path>`. When a type, `T`, implements the trait [`AsRef<U>`](https://doc.rust-lang.org/std/convert/trait.AsRef.html), it allows for cheap "implicit" casting from one reference type to the other. Rust is very strict with allowing implicit conversions, but the traits in the [std::convert](https://doc.rust-lang.org/std/convert/) crate can be used as bounds on generics, allowing for implicit conversion. In this case, any type that implements the `AsRef<Path>`, can be passed as an argument to the function `read_labels`. Conveniently, the Rust API reference has an ['Implementors'](https://doc.rust-lang.org/std/convert/trait.AsRef.html#implementors) section for each trait, which allows us to see which types implement this specific trait. In the case of this simple function, the addition of the generic is overkill, but it's instructive.

The first step in parsing the file is to open the file. In Rust, a file can be opened using the [std::fs::File](https://doc.rust-lang.org/std/fs/struct.File.html) struct. The open function takes a `Path` and returns an [`std::io::Result<File>`](https://doc.rust-lang.org/std/io/type.Result.html). This is similar to the [Maybe Monad](https://en.wikipedia.org/wiki/Monad_(functional_programming)#An_example:_Maybe) in Haskell and other functional languages. In Rust, one usually checks if the result is OK, and then unwraps the result (File), or returns the error attached to the `Result` struct. Rust included a convenient [question mark operator](https://doc.rust-lang.org/book/ch09-02-recoverable-errors-with-result.html?highlight=question,mark#propagating-errors) as a shortcut. This is the reason why the return type of our function is wrapped by a `std::io::Result`.

Continuing to the heart of the function, the reading of the data, we see that reading binary data in Rust is not that straightforward. We first need to create a slice to pass to the file reading function. Fortunately, you can create a slice from a pointer. Here I have to cast twice,
``` Rust
&mut magic_number as *mut u32 as *mut u8
```
But it would probably be more readable to write,
``` Rust
std::mem::transmute::<&mut u32, *mut u8>(&mut magic_number)
```
but I chose the former because it was shorter. At least we can use the methods implemented for `slice` to easily reverse the bytes. In a endian-independent implementation, we would extract the correct value like [here](https://commandcenter.blogspot.com/2012/04/byte-order-fallacy.html). The rest of the function is quite trivial.

We already see that doing low-level stuff in Rust is a bit more challenging. For comparison, here is the C++ implementation, which is endian-independent,
``` C++
static uint32_t be32_to_cpu(uint32_t x) {
    uint32_t ret = (
        ((uint32_t) ((uint8_t*)&x)[3] << 0)  |
        ((uint32_t) ((uint8_t*)&x)[2] << 8)  |
        ((uint32_t) ((uint8_t*)&x)[1] << 16) |
        ((uint32_t) ((uint8_t*)&x)[0] << 24)
    );
    return *(uint32_t*)(&ret);
}

void read_labels(
    const char* filepath,
    uint8_t** label_data, size_t* num_labels
)
{
    FILE* fp = fopen(filepath, "rb");

    uint32_t magic_number, num_items;

    fread(&magic_number, 4, 1, fp);
    magic_number = be32_to_cpu(magic_number);
    assert(magic_number == 0x801);

    fread(&num_items, 4, 1, fp);
    num_items = be32_to_cpu(num_items);

    *label_data = new uint8_t[num_items];
    fread(*label_data, num_items, 1, fp);

    fclose(fp);

    *num_labels = num_items;
}
```
As you can see, the C++ implementation of `read_labels` is less terse and shorter.

# Naive neural network implementation

For the neural network, I defined a `struct` for holding some important data for a layer,
``` Rust
struct NnLinearLayer
{
    input_size:  usize,
    output_size: usize,

    weights: Vec<f64>,
    biases:  Vec<f64>,
}
```
namely the biases and the weights, defined as a contiguous vector.
I also implemented two functions, one for constructing a layer, and one for processing inputs,
``` Rust
impl NnLinearLayer
{
    fn new(input_size: usize, output_size: usize) -> NnLinearLayer
    {
        let weights = thread_rng().sample_iter(&StandardNormal)
                        .take(input_size * output_size).collect();
        let biases  = thread_rng().sample_iter(&StandardNormal)
                        .take(output_size).collect();

        NnLinearLayer{
            input_size: input_size,
            output_size: output_size,
            weights: weights, biases: biases
        }
    }

    fn process_input(&self, input: &[f64]) -> Vec<f64>
    {
        assert!(input.len() == self.input_size);
        let mut output = self.biases.clone();
        for i in 0..self.output_size
        {
            for j in 0..self.input_size
            {
                output[i] += self.weights[i * self.input_size + j] *
                    input[j];
            }
        }
        output
    }
}
```
`process_input` simply does a matrix multiply of the input with the weights matrix, adding the bias. The constructor initializes the weights and biases with random numbers taken from the normal distribution. The random number generation crate is nice and quite easy to use. The implementation in C++ is very similar, with the only annoyance that you have to statically initialize the random engine. In Rust this is done for you by the crate.

The rest of the code, is mostly numerical calculations and the differences in implementations between C++ and Rust are minimal. I would say that my biggest annoyances in Rust are the fact that you are not allowed to have uninitialized memory in non-unsafe blocks. The functional nature of Rust's types also often promotes doing things in separate loops, although in a procedural implementation you would fuse the loops.

I trained a neural network with 23880 parameters for 30 epochs in batches of 10,
``` Rust
let mut layers = vec![
    NnLinearLayer::new(784, 30),
    NnLinearLayer::new(30, 10),
];
```
and the average training time per epoch on my MacBook Pro from mid 2014 was **14.742** seconds while the C++ version took an average **5.215** seconds per epoch. Both codes were built in release mode. The C++ version was compiled with `-march=native` and the Rust one with `-C target-cpu=native`.

I want to emphasize, here, that I'm **not** writing this blog to compare the performance between Rust and C++, so I'm not going to go down the rabbit hole, I'm sure someone can make both versions run as fast. This is more of a study of how easy it is to write low-level code in Rust and how performant it is on the first go.

# Applying SIMD optimizations

As I mentioned earlier, Rust did not support SIMD in stable until recently. Also, SIMD support was implemented differently in the [past](https://github.com/Grieverheart/dsfmt-rs/blob/master/src/mt19937.rs#L34). With the recent support, I wanted to test out how easy it is to write SIMD code in Rust. Before I dive-in, though, I'd like to mention that before implementing the SIMD optimizations, I thought I'd give a crate called [ndarray](https://crates.io/crates/ndarray) a try. As mentioned at the crate's page, `ndarray` implements an n-dimensional container for general elements and for numerics. `ndarray` can optionally use BLAS for better performance. Even though cargo makes the installation of crates very easy, it was not able to install BLAS properly on the Windows machine I started programming, so I gave up on that idea. Instead, I tried using BLAS kernels from C++, but performance was actually quite poor. I think this was due to an issue related to [this one](https://github.com/xianyi/OpenBLAS/issues/532).

So after a some failures, I decided to move on to the application of SIMD optimizations to the existing code. Vectorizing the neural network code should be straightforward as it is comprised of elementary vector operations matrix vector multiplications.

To be able to use the SIMD intrinsics in Rust, I have to import them like this,
``` Rust
use std::arch::x86_64::_mm256_loadu_ps;
```
Using the intrinsics from within Rust is actually pretty painless. Below you see the optimization of the `process_input` function,
``` Rust
fn process_input(&self, input: &[f32]) -> Vec<f32>
{
    assert!(input.len() == self.input_size);

    let mut output = Vec::with_capacity(self.output_size);
    let last_index = 8 * (self.input_size / 8);

    unsafe
    {
        output.set_len(self.output_size);

        for i in 0..self.output_size
        {
            let mut acc = _mm256_setzero_ps();
            for j in (0..last_index).step_by(8)
            {
                let sse_w  = _mm256_loadu_ps(
                    self.weights.as_ptr().offset(
                        (i * self.input_size + j) as isize
                    )
                );
                let sse_in = _mm256_loadu_ps(
                    input.as_ptr().offset(j as isize)
                );
                acc = _mm256_fmadd_ps(sse_w, sse_in, acc);
            }

            let temp_sum = _mm_add_ps(
                _mm256_extractf128_ps(acc, 1),
                _mm256_castps256_ps128(acc)
            );
            let temp_sum = _mm_add_ps(
                temp_sum,
                _mm_movehl_ps(temp_sum, temp_sum)
            );
            let temp_sum = _mm_add_ss(
                temp_sum,
                _mm_shuffle_ps(temp_sum, temp_sum, 0x55)
            );
            let sum = _mm_cvtss_f32(temp_sum);

            output[i] = self.biases[i] + sum;

            for j in last_index..self.input_size
            {
                output[i] +=
                    self.weights[i * self.input_size + j] * input[j];
            }
        }
    }

    output
}
```
Note that I change the type being processed to `f32` to allow for more vector operations. In this optimization, I naively split the loop into batches of 8, and perform the last remaining operations without using the intrinsics. A better idea would perhaps be to pad the data. Again, the C++ version is basically identical, but you can compare the two, [here](https://github.com/Grieverheart/neural_network).

With these optimizations applied to all numerical work, I saw big improvements in both implementations. The C++ version took an average of **1.155** seconds per epoch, a `4.5x` speedup -- there is still room for improvement, but that's not the goal of this blog post. The Rust version ran at an average **1.769** per epoch, that's a 8.3x speedup, but note that I also implemented some other optimizations, like preallocating the required space for the vectors that are used in the calculations. The Rust version did shorten the performance gap, with it being now only 53% slower.

# Conclusions

Rust feels like a really well designed language and not a mere superset of a language (although Mozilla made one big mistake, and that was naming dynamically sized arrays 'vectors'!). Although features have been borrowed from functional languages, they have been implemented in a way that really makes sense and do not feel out of place. The package manager is a truly amazing tool to have and promotes the sharing of code within the Rust community. Its borrow checker promotes moving variables instead of creating unnecessary (expensive) copies as is often the case in C++. Rust's documentation is also amazing, although it can be difficult to navigate for newcomers.

My experience with Rust in developing a neural network from scratch, is that low-level code in Rust is terse, and getting good performance is not that straightforward. Especially in a naive first implementation, I was able to get nearly 3 times better performance from C++ -- this was my experience 5 years ago too. On the other hand, the compiler errors are very helpful, and I did benefit from the array bounds checking when first developing the neural network. Clang and GCC do offer a bounds checking option nowadays via the `-fsanitize=address` compiler flag -- I feel this is a better option as to turn off bounds checking in Rust you have to use unchecked accesses, which, are unsafe, and thus need to be called from within an unsafe block, and are once again terser. I understand the motivation, and I like the idea of making only a certain part of the code unsafe, but I think it would help if the syntax was less terse and constrained, within these blocks.

From my current experimentation, Rust has not given me any compelling reasons to switch to it as my main language of choice. I really want to like Rust, I truly do. But I feel I might not be the target audience. Most of the time I don't need to write safe code. I think most people don't need to write safe code all the time. Rust is by default safe, making it hard to program most of the time so that certain bugs can be minimized. The question of course then is, does the effort spent writing safe code all the time outweigh the hypothetical reduction in bugs? Instead, with Moore's law slowly coming to an end, I think we need to design a system's programming language's that's fun to use, and more accessible to new programmers.
