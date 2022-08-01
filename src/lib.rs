use std::iter;
use std::thread;
use std::time::Duration;
use std::process;
use std::i16::{MIN,MAX};
use cpal::traits::HostTrait;
use itertools::Itertools;
use std::sync::{Arc, Mutex};
use cpal::{self, traits::DeviceTrait};

pub struct Encoder {
    device: cpal::Device,
    stream_config: cpal::StreamConfig,
    bit_length: usize,
}

impl Default for Encoder {
    fn default() -> Self {
        let host = cpal::default_host();
        let device = host.default_output_device().unwrap();
        let config = device.default_output_config().unwrap().into();

        Encoder::new(device, config, 1024)
    }
}

impl Encoder {
    pub fn new(device: cpal::Device, stream_config: cpal::StreamConfig, bitrate: usize) -> Encoder {
        assert_ne!(bitrate, 0, "Bitrate cannot be zero.");

        let channels = stream_config.channels as usize;
        let sample_rate = stream_config.sample_rate.0 as usize;
        let bit_length = channels * sample_rate / bitrate;

        Encoder {
            device,
            stream_config,
            bit_length,
        }
    }

    fn build_stream<T>(&self, data_callback: T) -> cpal::Stream
    where
        T: FnMut(&mut [i16], &cpal::OutputCallbackInfo) + Send + 'static
    {
        match self.device.build_output_stream(
            &self.stream_config,
            data_callback,
            |err| eprintln!("Stream error: {}", err)
        ) {
            Ok(stream) => stream,
            Err(err) => {
                eprintln!("Error creating stream: {}", err);
                process::exit(-1)
            }
        }
    }

    fn generate_header(&self, length: usize) -> Vec<u8> {
        let mut header = "OMEGALUL".as_bytes().to_vec();
        header.append(&mut length.to_be_bytes().to_vec());

        header
    }

    fn convert(&self, data: Vec<u8>) -> impl Iterator<Item = i16> {
        let bit_length = self.bit_length.clone();
        let header = self.generate_header(data.len());

        // Adding header and converting to (iterator over samples)
        header.into_iter()
            .chain(data.into_iter())
            .flat_map(|byte| {
                let mut bits = Vec::with_capacity(16);
                for i in 0..4 {
                    match byte << (2 * i) & 0b_1100_0000 {
                        0b_0000_0000 => {bits.push(MAX);                 bits.push(MIN)},
                        0b_0100_0000 => {bits.push(MAX);                 bits.push(MIN); bits.push(MIN)},
                        0b_1000_0000 => {bits.push(MAX); bits.push(MAX); bits.push(MIN)},
                        0b_1100_0000 => {bits.push(MAX); bits.push(MAX); bits.push(MIN); bits.push(MIN)},
                        _ => (),
                    }
                }
                bits
            })
            .flat_map(move |sample| iter::repeat(sample).take(bit_length))
    }

    pub fn send(&self, data: Vec<u8>) {
        let send_duration = Duration::from_secs((
            data.len() * 8 * self.bit_length / self.stream_config.sample_rate.0 as usize + 1)
            as u64);

        let samples = Mutex::new(self.convert(data));

        let _stream = self.build_stream(move |buffer: &mut [i16], _| {
            let mut samples = samples.lock().unwrap();
            for sample in buffer {
                *sample = samples.next().unwrap_or(0);
            }
        });

        thread::sleep(send_duration);


    }
}

const HEADER_BITS: [bool;64] = [
    false, true, false, false, true, true, true, true,
    false, true, false, false, true, true, false, true,
    false, true, false, false, false, true, false, true,
    false, true, false, false, false, true, true, true,
    false, true, false, false, false, false, false, true,
    false, true, false, false, true, true, false, false,
    false, true, false, true, false, true, false, true,
    false, true, false, false, true, true, false, false];

pub struct Decoder {
    device: cpal::Device,
    stream_config: cpal::StreamConfig,
    threshold: usize,
}

impl Default for Decoder {
    fn default() -> Self {
        let host = cpal::default_host();
        let device = host.default_input_device().unwrap();
        let config = device.default_input_config().unwrap().into();

        Decoder::new(device, config, 1024)
    }
}

impl Decoder {
    pub fn new(device: cpal::Device, stream_config: cpal::StreamConfig, bitrate: usize) -> Decoder {
        if bitrate == 0 {
            eprintln!("Error: Bitrate cannot be zero.");
            process::exit(-1)
        }

        let channels = stream_config.channels as usize;
        let sample_rate = stream_config.sample_rate.0 as usize;
        let threshold = 3 * channels * sample_rate / (2 * bitrate);

        Decoder {
            device,
            stream_config,
            threshold,
        }
    }

    fn build_stream<T>(&self, data_callback: T) -> cpal::Stream
    where
        T: FnMut(&[i16], &cpal::InputCallbackInfo) + Send + 'static
    {
        match self.device.build_input_stream(
            &self.stream_config,
            data_callback,
            |err| eprintln!("Stream error: {}", err)
        ) {
            Ok(stream) => stream,
            Err(err) => {
                eprintln!("Error creating stream: {}", err);
                process::exit(-1)
            }
        }
    }

    pub fn receive(&self) -> Vec<u8> {
        let mut residue = 0;
        let bit_buffer = Arc::new(Mutex::new(vec![]));
        let bit_buffer_2 = bit_buffer.clone();
        let threshold = self.threshold.clone();

        let _stream = self.build_stream(move |buffer: &[i16], _| {
            // Going from vec of samples to vec of (lengths of segments)
            let mut bits = buffer
                .iter()
                .map(|sample| if *sample > 0 {1 as isize} else {-1})
                .coalesce(|x, y| {
                    if (x >= 0) == (y >= 0) {
                        Ok(x + y)
                    } else {
                        Err((x, y))
                    }
                })
                .collect_vec();

            let residue_new = bits.pop().unwrap();
            
            if bits.is_empty() {
                residue += residue_new;
            }
            else {
                // filling bits to bit_buffer
                let mut bits = if (residue >= 0) == (*bits.first().unwrap() >= 0) {
                    *bits.first_mut().unwrap() += residue;
                    bits.into_iter()
                    .map(|bit| if bit.abs() as usize > threshold {true} else {false})
                    .collect()
                }
                else {
                    iter::once(residue)
                        .chain(bits.into_iter())
                        .map(|bit| if bit.abs() as usize > threshold {true} else {false})

                        .collect()
                };

                residue = residue_new;

                let mut bit_buffer = bit_buffer.lock().unwrap();
                bit_buffer.append(&mut bits);

            }
        });

        let byte_decoder = thread::spawn(move || {
            let mut header_found = false;
            let mut message_length = 0;
            let mut byte_buffer = vec![];

            'decoder: loop {
                thread::sleep(Duration::from_secs(1));

                let mut bit_buffer = bit_buffer_2.lock().unwrap();

                if !header_found {
                    // Looking for header
                    match bit_buffer
                        .windows(128)
                        .find_position(|header_window| header_window[..64] == HEADER_BITS[..])
                    {
                        Some((cutoff, _)) => {
                            bit_buffer.drain(..cutoff);

                            header_found = true;
                        },
                        None => {
                            let cutoff = bit_buffer.len() - 128;
                            bit_buffer.drain(..cutoff);
                        },
                    }
                }

                else {
                    // Draining the bit_buffer
                    let cutoff = bit_buffer.len() - bit_buffer.len() % 8;
                    let bits = bit_buffer.drain(..cutoff);
                    
                    // Going from vec of bits to vec of bytes
                    byte_buffer.extend(bits
                        .chunks(8)
                        .into_iter()
                        .map(|bits| {
                            let mut byte: u8 = 0;
                            for bit in bits {
                                byte <<= 1;
                                if bit {byte += 1}
                            }
                            byte
                        })
                    );

                    drop(bit_buffer);

                    // Setting final lengths
                    if message_length == 0 {
                        message_length = u64::from_be_bytes(byte_buffer[8..16]
                            .try_into().unwrap()) as usize;
                        let mut residue = byte_buffer[16..].to_vec();
                        byte_buffer = Vec::with_capacity(message_length);
                        byte_buffer.append(&mut residue);
                    }

                    // Checking whether message is complete
                    if byte_buffer.len() >= message_length {
                        let message: Vec<_> = byte_buffer
                            .iter()
                            .take(message_length)
                            .map(|x| *x)
                            .collect();
                        break 'decoder message;
                    }
                }
            }
        });

        byte_decoder.join().unwrap()


    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    #[ignore = "requires default audio output to be bridged with default audio input"]
    fn encode_and_decode() {
        let decoder = Decoder::default();
        let encoder = Encoder::default();

        let data = [1,2,3,4,5].to_vec();
        let data2 = data.clone();

        thread::spawn(move || {
            thread::sleep(Duration::from_secs(1));
            encoder.send(data);
        });

        assert_eq!(data2, decoder.receive());
    }
}