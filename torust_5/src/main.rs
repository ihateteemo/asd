#![feature(linked_list_remove)]
use rand::{distributions::{Distribution, Bernoulli}, prelude::*};
use std::{collections::VecDeque, path::PathBuf};
use std::fs;
use std::thread;
use std::collections::LinkedList;

fn argmax<T: PartialOrd + Copy> (vec: &Vec<T>) -> usize {
    let mut max = vec[0];
    let mut idx: usize = 0;

    for index in 0..vec.len() {
        if vec[index] > max {
            max = vec[index];
            idx = index;
        }
    }
    idx
}

fn argmin<T: PartialOrd + Copy> (vec: &Vec<T>) -> usize {
    let mut min = vec[0];
    let mut idx: usize = 0;

    for index in 0..vec.len() {
        if vec[index] < min {
            min = vec[index];
            idx = index;
        }
    }
    idx
}

fn idx<T: PartialOrd + Copy> (vec: &Vec<T>, val: &T) -> usize {
    for index in 0..vec.len() {
        if vec[index] == *val {
            return index
        }
    }
    0
}

fn pow (val0: &f32, val1: &usize) -> f32 {
    if *val1 == 0 {
        1.0
    }
    else {
        let mut body = *val0;
    
        for _ in 0..*val1 -1 {
            body *= val0;
        }
        body
    }
}

fn sum_deq (deq: &VecDeque<f32>) -> f32 {
    let mut sum: f32 = 0.0;
    
    for val in deq {
        sum += *val;
    }
    sum
}

fn min<T: Copy + PartialOrd> (vec: &Vec<T>) -> T {
    let mut min: T = vec[0];
    for val in vec {
        if min > *val {
            min = *val;
        }
    }
    min
}

fn max<T: Copy + PartialOrd> (vec: &Vec<T>) -> T {
    let mut max: T = vec[0];
    for val in vec {
        if max < *val {
            max = *val;
        }
    }
    max
}

fn mean_f (vec: &Vec<f32>) -> f32 {
    let mut sum: f32 = 0.0;
    let len: f32 = vec.len() as f32;
    for val in vec {
        sum += *val;
    }
    let mean = sum / len;
    mean
}

fn mean_usize (vec: &Vec<usize>) -> f32 {
    let mut sum: usize = 0;
    let len: f32 = vec.len() as f32;
    for val in vec {
        sum += *val;
    }
    let sum = sum as f32;
    sum / len
}

struct Neuron {
    name: usize,
    name_dend: usize,
    name_axon: usize,
    weight_type: usize,
    output_type: usize,
    index_dendbody: usize,
    index_axonbody: usize,
    index_inp: usize,
    index_axonbranch: Vec<usize>,
    index_dendbranch: Vec<usize>,
    score: Vec<f32>,
    new_axonbranch: Vec<usize>
}

fn init_neuron (
    num_weight_type: &usize,
    num_output_type: &usize,
    num_space: &usize,
    input_unit: &usize
) -> Neuron {
    let mut rng: ThreadRng = rand::thread_rng();
    let neuron_weight_type: usize = rng.gen_range(0..*num_weight_type);
    let neuron_output_type: usize = rng.gen_range(0..*num_output_type);
    
    Neuron {
        name: 0,
        name_dend: rng.gen_range(0..2),
        name_axon: rng.gen_range(0..3),
        weight_type: neuron_weight_type,
        output_type: neuron_output_type,
        index_dendbody: rng.gen_range(0..*num_space),
        index_axonbody: rng.gen_range(0..*num_space),
        index_inp: rng.gen_range(0..*input_unit),
        index_axonbranch: Vec::new(),
        index_dendbranch: Vec::new(),
        score: Vec::new(),
        new_axonbranch: Vec::new()
    }
}

impl Neuron {
    fn fire (
        &self,
        neuron_output_types: &Vec<usize>,
        neuron_outs: &Vec<f32>,
        neuron_num_output_type: &usize,
        neuron_space: &Vec<Vec<f32>>,
        neuron_threshold: &f32,
        neuron_data: Vec<f32>,
        neuron_out_max: &f32,
        neuron_weight_cal: &Vec<Vec<f32>>
    ) -> f32 {
        let mut out: f32 = 0.0;

        if self.name_dend == 0 {
            self.index_dendbranch.iter().for_each(|index_dend: &usize| {
                out += neuron_weight_cal[self.weight_type][neuron_output_types[*index_dend]] * neuron_outs[*index_dend];
            });

            (0..*neuron_num_output_type).for_each(|outtypes| {
                out += neuron_weight_cal[self.weight_type][outtypes] * neuron_space[self.index_dendbody][outtypes];
            });
            
            out = if out < 0.0 {
                0.0
            }
            else {
                out /= neuron_threshold;
                out.floor()
            };
        }
        else {
            out += neuron_data[self.index_inp];
        }

        if *neuron_out_max < out {
            out = *neuron_out_max;
        }
        
        out
    }

    fn dendpruning (
        &mut self,
        neuron_num_output_type: &usize,
        neuron_output_types: &Vec<usize>,
        neuron_outs: &Vec<f32>,
        neuron_space: &Vec<Vec<f32>>,
        neuron_score_max: &f32,
        neuron_fixed_axonbranch: &Vec<Vec<usize>>,
        neuron_score_init: &f32,
        neuron_weight_pru: &Vec<Vec<Vec<f32>>>
    ) -> Vec<usize> {
        let mut pru: Vec<f32> = Vec::new();

        (0..*neuron_num_output_type).for_each(|_| {
            pru.push(0.0);
        });

        self.index_dendbranch.iter().for_each(|index_dend| {
            (0..*neuron_num_output_type).for_each(|index_pru| {
                pru[index_pru] += neuron_weight_pru[self.weight_type][neuron_output_types[*index_dend]][index_pru] * neuron_outs[*index_dend];
            });
        });
        (0..*neuron_num_output_type).for_each(|outtypes| {
            (0..*neuron_num_output_type).for_each(|index_pru| {
                pru[index_pru] += neuron_weight_pru[self.weight_type][outtypes][index_pru] * neuron_space[self.index_dendbody][outtypes];
            });
        });
        
        for (index, outtypes) in self.index_dendbranch.iter().enumerate() {
            self.score[index] += pru[neuron_output_types[*outtypes]];
        }

        let mut disconnected: Vec<usize> = Vec::new();
        let mut index: usize = 0;
        let mut len: usize = self.score.len();

        loop {
            if len == index {
                break;
            }
            if self.score[index] > *neuron_score_max {
                self.score[index] = *neuron_score_max;
            }
            else if self.score[index] <= 0.0 {
                disconnected.push(self.index_dendbranch[index]);
                self.score.swap_remove(index);
                self.index_dendbranch.swap_remove(index);
                len -= 1;
                continue;
            }
            else {
            }
            index += 1;
        }

        for (index_axon, list_index_dend) in neuron_fixed_axonbranch.iter().enumerate() {
            list_index_dend.into_iter().for_each(|index_dend: &usize| {
                if self.name == *index_dend {
                    self.index_dendbranch.push(index_axon);
                    self.score.push(*neuron_score_init);
                }
            });
        }

        disconnected
    }

    fn axonpruning (
        &mut self,
        neuron_disconnected: &Vec<Vec<usize>>,
        neuron_space: &Vec<Vec<f32>>,
        neuron_num_output_type: &usize,
        neuron_index_cal: &Vec<Vec<usize>>,
        neuron_value_cal: &Vec<Vec<f32>>,
        neuron_num_space: &usize,
        neuron_num_axonbranch: &usize
    ) -> Vec<usize> {

        for (index_dend, list_index_axon) in neuron_disconnected.iter().enumerate() {
            list_index_axon.into_iter().for_each(|index_axon| {
                if self.name == *index_axon {
                    let del = idx(&self.index_axonbranch, &index_dend);
                    self.index_axonbranch.swap_remove(del);
                }
            });
        }

        let mut index: usize = 0;
        let mut len: usize = self.new_axonbranch.len();
        let mut fixed_axonbranch: Vec<usize> = Vec::new();

        loop {
            if len == index {
                break;
            }
            let new = self.new_axonbranch[index];
            if new == 0 {
                if neuron_space[new][*neuron_num_output_type] > neuron_space[new + 1][*neuron_num_output_type] {
                    if neuron_index_cal[new].len() > 0 {
                        let connected: usize = neuron_index_cal[new][argmax(&neuron_value_cal[new])];
                        fixed_axonbranch.push(connected);
                        self.index_axonbranch.push(connected);
                        self.new_axonbranch.swap_remove(index);
                        len -= 1;
                        continue;
                    }
                    else {
                        self.new_axonbranch[index] += 1;
                    }
                }
            }
            else if new == *neuron_num_space - 1 {
                if neuron_space[new][*neuron_num_output_type] > neuron_space[new-1][*neuron_num_output_type] {
                    if neuron_index_cal[new].len() > 0 {
                        let connected: usize = neuron_index_cal[new][argmax(&neuron_value_cal[new])];
                        fixed_axonbranch.push(connected);
                        self.index_axonbranch.push(connected);
                        self.new_axonbranch.swap_remove(index);
                        len -= 1;
                        continue;
                    }
                    else {
                        self.new_axonbranch[index] -= 1;
                    }
                }
            }
            else {
                if neuron_space[new-1][*neuron_num_output_type] > neuron_space[new][*neuron_num_output_type] && neuron_space[new-1][*neuron_num_output_type] > neuron_space[new+1][*neuron_num_output_type] {
                    self.new_axonbranch[index] -= 1;
                }
                else if neuron_space[new+1][*neuron_num_output_type] > neuron_space[new][*neuron_num_output_type] && neuron_space[new+1][*neuron_num_output_type] > neuron_space[new-1][*neuron_num_output_type] {
                    self.new_axonbranch[index] += 1;
                }
                else if neuron_space[new][*neuron_num_output_type] > neuron_space[new-1][*neuron_num_output_type] && neuron_space[new][*neuron_num_output_type] > neuron_space[new+1][*neuron_num_output_type] {
                    if neuron_index_cal[new].len() > 0 {
                        let connected: usize = neuron_index_cal[new][argmax(&neuron_value_cal[new])];
                        fixed_axonbranch.push(connected);
                        self.index_axonbranch.push(connected);
                        self.new_axonbranch.swap_remove(index);
                        len -= 1;
                        continue;
                    }
                }
                else {
                }
            }
            index += 1;
        }

        if self.index_axonbranch.len() + self.new_axonbranch.len() < *neuron_num_axonbranch {
            self.new_axonbranch.push(self.index_axonbody);
        }

        fixed_axonbranch
    }

    fn mutation (
        &mut self,
        neuron_bernoulli: &Bernoulli,
        neuron_num_weight_type: &usize,
        neuron_num_output_type: &usize,
        neuron_num_space: &usize,
        neuron_input_unit: &usize,
        neuron_rng: &mut ThreadRng
    ) {
        if neuron_bernoulli.sample(neuron_rng) == true {
            self.name_dend = neuron_rng.gen_range(0 .. 2);
            
        }
        if neuron_bernoulli.sample(neuron_rng) == true {
            self.name_axon = neuron_rng.gen_range(0 .. 3);
        }
        if neuron_bernoulli.sample(neuron_rng) == true {
            self.weight_type = ((self.weight_type as i32) + neuron_rng.gen_range(-1 .. 2)) as usize;
            if self.weight_type >= *neuron_num_weight_type {
                self.weight_type = *neuron_num_weight_type - 1;
            }
        }
        if neuron_bernoulli.sample(neuron_rng) == true {
            self.output_type = ((self.output_type as i32) + neuron_rng.gen_range(-1 .. 2)) as usize;
            if self.output_type >= *neuron_num_output_type {
                self.output_type = *neuron_num_output_type - 1;
            }
        }
        if neuron_bernoulli.sample(neuron_rng) == true {
            if self.index_dendbody == 0 {
                self.index_dendbody += 1;
            }
            else {
                self.index_dendbody = ((self.index_dendbody as i32) + neuron_rng.gen_range(-1 .. 2)) as usize;
                if self.index_dendbody >= *neuron_num_space {
                    self.index_dendbody = *neuron_num_space - 1;
                }
            }
        }
        if neuron_bernoulli.sample(neuron_rng) == true {
            if self.index_axonbody == 0 {
                self.index_axonbody += 1;
            }
            else {
                self.index_axonbody = ((self.index_axonbody as i32) + neuron_rng.gen_range(-1 .. 2)) as usize;
                if self.index_axonbody >= *neuron_num_space {
                    self.index_axonbody = *neuron_num_space - 1;
                }
            }
        }
        if neuron_bernoulli.sample(neuron_rng) == true {
            if self.index_inp == 0 {
                self.index_inp += 1
            }
            else {
                self.index_inp = ((self.index_inp as i32) + neuron_rng.gen_range(-1 .. 2)) as usize;
                if self.index_inp >= *neuron_input_unit {
                    self.index_inp = *neuron_input_unit - 1;
                }
            }
        }
    }
}
impl Clone for Neuron {
    fn clone (&self) -> Neuron {
        Neuron {
            name: self.name,
            name_dend: self.name_dend,
            name_axon: self.name_axon,
            weight_type: self.weight_type,
            output_type: self.output_type,
            index_dendbody: self.index_dendbody,
            index_axonbody: self.index_axonbody,
            index_inp: self.index_inp,
            index_axonbranch: self.index_axonbranch.clone(),
            index_dendbranch: self.index_dendbranch.clone(),
            score: self.score.clone(),
            new_axonbranch: self.new_axonbranch.clone()
        }
    }
}

struct Individual {
    name: usize,
    threshold_out: f32,
    score: f32,
    num_weight_type: usize,
    num_output_type: usize,
    num_space: usize,
    num_neuron: usize,
    num_axonbranch: usize,
    threshold: f32,
    input_unit: usize,
    reuptake: f32,
    score_init: f32,
    score_max: f32,
    weight_cal: Vec<Vec<f32>>,
    weight_pru: Vec<Vec<Vec<f32>>>,
    output_types: Vec<usize>,
    outs: Vec<f32>,
    space: Vec<Vec<f32>>,
    disconnected: Vec<Vec<usize>>,
    fixed_axonbranch: Vec<Vec<usize>>,
    index_cal: Vec<Vec<usize>>,
    value_cal: Vec<Vec<f32>>,
    neurons: Vec<Neuron>,
    con: Vec<usize>,
    age: usize,
    adult: bool,
    threshold_adult: usize,
    out_max: f32,
    scores: VecDeque<f32>,
    deq_size: usize,
    deq_filled: bool,
    idv_score_init: f32
}

fn init_individual (
    num_weight_type: &usize,
    num_output_type: &usize,
    num_space: &usize,
    num_neuron: &usize,
    num_axonbranch: &usize,
    threshold: &f32,
    input_unit: &usize,
    reuptake: &f32,
    score_init: &f32,
    score_max: &f32,
    threshold_out: &f32,
    idv_score_init: &f32,
    threshold_adult: &usize,
    out_max: &f32,
    deq_size: &usize
) -> Individual {
    let mut rng = rand::thread_rng();

    let mut weight_cal: Vec<Vec<f32>> = Vec::new();

    (0 .. *num_weight_type).for_each(|_| {
        let mut vec0: Vec<f32> = Vec::new();
        (0 .. *num_output_type).for_each(|_| {
            vec0.push(rng.gen_range(-1.0 .. 1.0));
        });
        weight_cal.push(vec0);
    });

    let mut weight_pru: Vec<Vec<Vec<f32>>> = Vec::new();

    (0 .. *num_weight_type).for_each(|_| {
        let mut vec0: Vec<Vec<f32>> = Vec::new();
        (0 .. *num_output_type).for_each(|_| {
            let mut vec1: Vec<f32> = Vec::new();
            (0 .. *num_output_type).for_each(|_| {
                vec1.push(rng.gen_range(-1.0 .. 1.0));
            });
            vec0.push(vec1);
        });
        weight_pru.push(vec0);
    });

    let mut output_types: Vec<usize> = Vec::new();
    let mut outs: Vec<f32> = Vec::new();

    (0 .. *num_neuron).for_each(|_| {
        output_types.push(0);
        outs.push(0.0);
    });

    let mut space: Vec<Vec<f32>> = Vec::new();

    (0 .. *num_space).for_each(|_| {
        let mut vec0: Vec<f32> = Vec::new();
        (0 .. (*num_output_type + 1)).for_each(|_| {
            vec0.push(0.0);
        });
        space.push(vec0);
    });

    let mut disconnected: Vec<Vec<usize>> = Vec::new();
    let mut fixed_axonbranch: Vec<Vec<usize>> = Vec::new();
    let mut index_cal: Vec<Vec<usize>> = Vec::new();
    let mut value_cal: Vec<Vec<f32>> = Vec::new();

    (0 .. *num_neuron).for_each(|_| {
        let vec0: Vec<usize> = Vec::new();
        disconnected.push(vec0.clone());
        fixed_axonbranch.push(vec0.clone());
    });
    (0 .. *num_space).for_each(|_| {
        let vec0: Vec<usize> = Vec::new();
        let vec1: Vec<f32> = Vec::new();
        index_cal.push(vec0);
        value_cal.push(vec1);
    });

    let mut neurons: Vec<Neuron> = Vec::new();

    (0 .. *num_neuron).for_each(|index| {
        let new_neuron: Neuron = init_neuron(
            num_weight_type,
            num_output_type,
            num_space,
            input_unit,
        );
        neurons.push(new_neuron);
        neurons[index].name = index;
    });

    let mut con: Vec<usize> = Vec::new();

    neurons.iter().for_each(|n| {
        if n.name_axon == 0 {
            con.push(n.name);
        }
    });

    for (index, n) in neurons.iter().enumerate() {
        output_types[index] = n.output_type;
        
        if n.name_dend == 1 {
            index_cal[n.index_dendbody].push(index);
        }
    }

    for (index_space, list_name_neuron) in index_cal.iter().enumerate() {
        list_name_neuron.into_iter().for_each(|_| {
            value_cal[index_space].push(0.0);
        });
    }

    let scores: VecDeque<f32> = VecDeque::new();

    Individual {
        name: 0,
        threshold_out: *threshold_out,
        score: *idv_score_init,
        num_weight_type: *num_weight_type,
        num_output_type: *num_output_type,
        num_space: *num_space,
        num_neuron: *num_neuron,
        num_axonbranch: *num_axonbranch,
        threshold: *threshold,
        input_unit: *input_unit,
        reuptake: *reuptake,
        score_init: *score_init,
        score_max: *score_max,
        weight_cal: weight_cal,
        weight_pru: weight_pru,
        output_types: output_types,
        outs: outs,
        space: space,
        disconnected: disconnected,
        fixed_axonbranch: fixed_axonbranch,
        index_cal: index_cal,
        value_cal: value_cal,
        neurons: neurons,
        con: con,
        age: 0,
        adult: false,
        threshold_adult: *threshold_adult,
        out_max: *out_max,
        scores: scores,
        deq_size: *deq_size,
        deq_filled: false,
        idv_score_init: *idv_score_init
    }
}

impl Individual {
    fn action (
        &mut self,
        individual_data: &Vec<f32>,
        individual_penalty: &f32,
        answer: &usize,
    ) {
        let epoch: usize = individual_data.len() / self.input_unit;
        let input_unit = self.input_unit;

        (0..(epoch + 10)).for_each(|e| {
            
            let data = |e: usize, epoch: usize, individual_data: &&Vec<f32>| {
                if e < epoch {
                    individual_data[e * (input_unit) .. (e+1) * (input_unit)].to_vec()
                }
                else if e == epoch {
                    let mut data0 = individual_data[e * input_unit ..].to_vec();
                    (0..(input_unit - data0.len())).for_each(|_| {
                        data0.push(0.0);
                    });
                    data0
                }
                else {
                    let mut data1: Vec<f32> = Vec::new();
                    (0..input_unit).for_each(|_| {
                        data1.push(0.0);
                    });
                    data1
                }
            };

            let mut fixed_axonbranch_t2 = vec![Vec::new(); self.fixed_axonbranch.len()];
            let mut disconnected_t2 = vec![Vec::new(); self.disconnected.len()];
            let mut space_t2 = self.space.clone();
            let mut value_cal_t2 = self.value_cal.clone();
            let mut outs_t2 = vec![f32::default(); self.outs.len()];
        
            for n in self.neurons.iter_mut() {
                outs_t2[n.name] = n.fire(
                    &self.output_types,
                    &self.outs,
                    &self.num_output_type,
                    &self.space,
                    &self.threshold,
                    data(e, epoch, &individual_data),
                    &self.out_max,
                    &self.weight_cal
                );

                if n.name_dend == 1 {
                    space_t2[n.index_dendbody][self.num_output_type] += outs_t2[n.name];
                    value_cal_t2[n.index_dendbody][idx(&self.index_cal[n.index_axonbody], &n.name)] = outs_t2[n.name];

                    disconnected_t2[n.name] = n.dendpruning(
                        &self.num_output_type,
                        &self.output_types,
                        &self.outs,
                        &self.space,
                        &self.score_max,
                        &self.fixed_axonbranch,
                        &self.score_init,
                        &self.weight_pru
                    );
                }
                if n.name_axon == 1 {
                    space_t2[n.index_axonbody][n.output_type] += outs_t2[n.name];
                }
                else if n.name_axon == 2 {
                    fixed_axonbranch_t2[n.name] = n.axonpruning(
                        &self.disconnected,
                        &self.space,
                        &self.num_output_type,
                        &self.index_cal,
                        &self.value_cal,
                        &self.num_space,
                        &self.num_axonbranch
                    );
                }
            }

            self.value_cal.iter().zip(space_t2.iter_mut()).for_each(|(vc,s)| {
                s.iter_mut().for_each(|s2| {
                    *s2 *= pow(&self.reuptake, &vc.len());
                });
            });

            (0..self.num_space).for_each(|index_c| {
                (0..(self.num_output_type+1)).for_each(|index_r| {
                    if index_c == 0 {
                        self.space[index_c][index_r] = (space_t2[index_c][index_r] + space_t2[index_c+1][index_r])/2.0;
                    }
                    else if index_c + 1 == self.num_space {
                        self.space[index_c][index_r] = (space_t2[index_c][index_r] + space_t2[index_c-1][index_r])/2.0;
                    }
                    else {
                        self.space[index_c][index_r] = (space_t2[index_c-1][index_r] + space_t2[index_c][index_r] + space_t2[index_c+1][index_r])/3.0;
                    }
                });
            });

            self.disconnected = disconnected_t2;
            self.fixed_axonbranch = fixed_axonbranch_t2;
            self.outs = outs_t2;
            self.value_cal = value_cal_t2;
        });
        
        let mut out: f32 = 0.0;
        
        self.con.iter().for_each(|index| {
            out += self.outs[*index];
        });

        let mut y: usize = if out > self.threshold_out {
            0
        }
        else {
            1
        };

        if self.con.len() == 0 {
            y = 2;
        }
        
        if self.deq_filled == false {
            if self.scores.len() == self.deq_size {
                self.deq_filled = true;
            }
        }

        if self.deq_filled == true {
            self.scores.pop_front();
        }
        
        if y == *answer {
            self.scores.push_back(1.0);
        }
        else {
            self.scores.push_back(*individual_penalty * -1.0);
        }
        
        self.score = self.idv_score_init + sum_deq(&self.scores);

        self.age += 1;
        if self.adult == false {
            if self.age == self.threshold_adult {
                self.adult = true;
            }
        }
    }

    fn mutation (
        &mut self,
        individual_bernoulli: &Bernoulli,
    ) {
        let prev: [usize; 5] = [self.num_weight_type.clone(), self.num_output_type.clone(), self.num_space.clone(), self.num_neuron.clone(), self.input_unit.clone()];
        let mut rng = thread_rng();

        if individual_bernoulli.sample(&mut rng) == true {
            self.num_weight_type = ((self.num_weight_type as i32) + rng.gen_range(-1 .. 2)) as usize;
            if self.num_weight_type < 10 {
                self.num_weight_type = 10;
            }
        }

        if individual_bernoulli.sample(&mut rng) == true {
            self.num_output_type = ((self.num_output_type as i32) + rng.gen_range(-1 .. 2)) as usize;
            if self.num_output_type < 10 {
                self.num_output_type = 10;
            }
        }

        if individual_bernoulli.sample(&mut rng) == true {
            self.num_space = ((self.num_space as i32) + rng.gen_range(-1 .. 2)) as usize;
            if self.num_space < 100 {
                self.num_space = 100;
            }
        }
        
        if individual_bernoulli.sample(&mut rng) == true {
            self.num_neuron = ((self.num_neuron as i32) + rng.gen_range(-1 .. 2)) as usize;
            if self.num_neuron < 10 {
                self.num_neuron = 10;
            }
        }
        
        if individual_bernoulli.sample(&mut rng) == true {
            self.num_axonbranch = ((self.num_axonbranch as i32) + rng.gen_range(-1 .. 2)) as usize;
            if self.num_axonbranch < 10 {
                self.num_axonbranch = 10;
            }
        }
        
        if individual_bernoulli.sample(&mut rng) == true {
            self.input_unit = ((self.input_unit as i32) + rng.gen_range(-1 .. 2)) as usize;
            if self.input_unit < 10 {
                self.input_unit = 10;
            }
        }
        
        if individual_bernoulli.sample(&mut rng) == true {
            self.threshold_adult = ((self.threshold_adult as i32) + rng.gen_range(-1 .. 2)) as usize;
            if self.threshold_adult < 20 {
                self.threshold_adult = 20;
            }
        }
        
        if individual_bernoulli.sample(&mut rng) == true {
            self.deq_size = ((self.deq_size as i32) + rng.gen_range(-1 .. 2)) as usize;
            if self.deq_size < 20 {
                self.deq_size = 20;
            }
        }
    
        self.threshold += rng.gen_range(-0.03 .. 0.03);
        if self.threshold < 10.0 {
            self.threshold = 10.0;
        }

        self.threshold_out += rng.gen_range(-0.03 .. 0.03);
        if self.threshold_out < 20.0 {
            self.threshold_out = 20.0;
        }
    
        self.score_init += rng.gen_range(-0.03 .. 0.03);
        if self.score_init < 10.0 {
            self.score_init = 10.0;
        }

        self.score_max += rng.gen_range(-0.03 .. 0.03);
        if self.score_max < 100.0 {
            self.score_max = 100.0;
        }
    
        self.reuptake += rng.gen_range(-0.0003 .. 0.0003);
        if self.reuptake > 1.0 {
            self.reuptake = 1.0;
        }
        else if self.reuptake < 0.5 {
            self.reuptake = 0.5;
        }

        self.weight_cal.iter_mut().for_each(|f| {
            f.iter_mut().for_each(|f2| {
                *f2 += rng.gen_range(-0.03 .. 0.03);
            });
        });

        self.weight_pru.iter_mut().for_each(|f| {
            f.iter_mut().for_each(|f2| {
                f2.iter_mut().for_each(|f3| {
                    *f3 += rng.gen_range(-0.03 .. 0.03);
                });
            });
        });

        if prev[0] < self.num_weight_type {
            let duplicated: usize = rng.gen_range(0 .. self.weight_cal.len());
            
            self.weight_cal.insert(duplicated, self.weight_cal[duplicated].clone());

            self.weight_pru.insert(duplicated, self.weight_pru[duplicated].clone());

            self.neurons.iter_mut().for_each(|n| {
                if n.weight_type > duplicated {
                    n.weight_type += 1;
                }
            });
        }

        if prev[1] < self.num_output_type {
            let duplicated: usize = rng.gen_range(0 .. self.weight_cal[0].len());

            self.weight_cal.iter_mut().for_each(|vec| {
                vec.insert(duplicated, vec[duplicated]);
            });

            self.weight_pru.iter_mut().for_each(|vec_2d| {
                
                vec_2d.into_iter().for_each(|vec_1d| {
                    vec_1d.insert(duplicated, vec_1d[duplicated]);
                });
            });

            self.weight_pru.iter_mut().for_each(|vec_2d| {
                vec_2d.insert(duplicated,vec_2d[duplicated].clone());
            });

            self.neurons.iter_mut().for_each(|n| {
                if n.output_type > duplicated {
                    n.output_type += 1;
                }
            });
            (0 .. self.space.len()).for_each(|index| {
                self.space[index].push(0.0);
            });
        }

        if prev[0] > self.num_weight_type {
            let deleted: usize = rng.gen_range(0 .. self.weight_cal.len());

            self.weight_cal.remove(deleted);
            self.weight_pru.remove(deleted);

            for n in self.neurons.iter_mut() {
                if n.weight_type > deleted {
                    n.weight_type -= 1;
                }
                else if n.weight_type == deleted {
                    n.weight_type = ((n.weight_type as i32) + rng.gen_range(-1 .. 2)) as usize;
                    if n.weight_type >= self.num_weight_type {
                        n.weight_type = self.num_weight_type - 1;
                    }
                }
            }
        }

        if prev[1] > self.num_output_type {
            let deleted: usize = rng.gen_range(0 .. self.weight_cal[0].len());

            self.weight_cal.iter_mut().for_each(|vec_1d| {
                vec_1d.remove(deleted);
            });

            self.weight_pru.iter_mut().for_each(|vec_2d| {
                vec_2d.remove(deleted);
                vec_2d.into_iter().for_each(|vec_1d| {
                    vec_1d.remove(deleted);
                });
            });

            for n in self.neurons.iter_mut() {
                if n.output_type > deleted {
                    n.output_type -= 1;
                }
                else if n.output_type == deleted {
                    n.output_type = ((n.output_type as i32) + rng.gen_range(-1 .. 2)) as usize;
                    if n.output_type >= self.num_output_type {
                        n.output_type = self.num_output_type - 1;
                    }
                }
            }

            (0..self.space.len()).for_each(|index| {
                self.space[index].pop();
            });
        }

        if prev[2] > self.num_space {
            for n in self.neurons.iter_mut() {
                if n.index_axonbody == self.num_space {
                    n.index_axonbody -= 1;
                }
                if n.index_dendbody == self.num_space {
                    n.index_dendbody -= 1;
                }
            }
            self.space.pop();
        }

        if prev[2] < self.num_space {
            let new = self.space[0].clone();
            self.space.push(new);
        }

        if prev[3] > self.num_neuron {
            let deleted: usize = rng.gen_range(0..prev[3]);
            self.neurons.remove(deleted);

            self.output_types.pop();
            self.outs.pop();
            self.disconnected.pop();
            self.fixed_axonbranch.pop();
        }

        if prev[4] > self.input_unit {
            let deleted: usize = rng.gen_range(0..prev[4]);
            for n in self.neurons.iter_mut() {
                if n.index_inp > deleted {
                    n.index_inp -= 1;
                }
                else if n.index_inp == deleted {
                    n.index_inp = ((n.index_inp as i32) + rng.gen_range(-1 .. 2)) as usize;
                    if n.index_inp >= self.input_unit {
                        n.index_inp = self.input_unit - 1;
                    }
                }
            }
        }

        for n in &mut self.neurons {

            n.mutation(
                &individual_bernoulli,
                &self.num_weight_type,
                &self.num_output_type,
                &self.num_space,
                &self.input_unit,
                &mut rng
            )
        }
        
        if prev[3] < self.num_neuron {
            let new_neuron: Neuron = init_neuron(
                &self.num_weight_type,
                &self.num_output_type,
                &self.num_space,
                &self.input_unit
            );

            self.neurons.push(new_neuron);

            let x: Vec<usize> = Vec::new();
            self.output_types.push(0);
            self.outs.push(0.0);
            self.disconnected.push(x.clone());
            self.fixed_axonbranch.push(x.clone());
        }

        let mut index: usize = 0;

        for n in &mut self.neurons {
            n.name = index;
            index += 1;
        }

        let mut new_con: Vec<usize> = Vec::new();

        for n in &self.neurons {
            if n.name_axon == 0 {
                new_con.push(n.name);
            }
        }

        self.con = new_con;

    
        let mut new_index_cal: Vec<Vec<usize>> = Vec::new();
        let mut new_value_cal: Vec<Vec<f32>> = Vec::new();

        (0..self.num_space).for_each(|_| {
            let vec0: Vec<usize> = Vec::new();
            let vec1: Vec<f32> = Vec::new();

            new_index_cal.push(vec0);
            new_value_cal.push(vec1);
        });
        
        for (index, n) in self.neurons.iter().enumerate() {
            self.output_types[index] = n.output_type;
            
            if n.name_dend == 1 {
                new_index_cal[n.index_dendbody].push(index);
            }
        }
    
        for (index_space, list_name_neuron) in new_index_cal.iter().enumerate() {
            (0..(list_name_neuron.len())).for_each(|_| {
                new_value_cal[index_space].push(0.0);
            });
        }

        self.index_cal = new_index_cal;
        self.value_cal = new_value_cal;

        self.scores = VecDeque::new();
    }
}

impl Clone for Individual {
    fn clone(&self) -> Individual {
        Individual {
            name: self.name,
            threshold_out: self.threshold_out,
            score: self.score,
            num_weight_type: self.num_weight_type,
            num_output_type: self.num_output_type,
            num_space: self.num_space,
            num_neuron: self.num_neuron,
            num_axonbranch: self.num_axonbranch,
            threshold: self.threshold,
            input_unit: self.input_unit,
            reuptake: self.reuptake,
            score_init: self.score_init,
            score_max: self.score_max,
            weight_cal: self.weight_cal.clone(),
            weight_pru: self.weight_pru.clone(),
            output_types: self.output_types.clone(),
            outs: self.outs.clone(),
            space: self.space.clone(),
            disconnected: self.disconnected.clone(),
            fixed_axonbranch: self.fixed_axonbranch.clone(),
            index_cal: self.index_cal.clone(),
            value_cal: self.value_cal.clone(),
            neurons: self.neurons.clone(),
            con: self.con.clone(),
            age: self.age,
            adult: self.adult,
            threshold_adult: self.threshold_adult,
            out_max: self.out_max,
            scores: self.scores.clone(),
            deq_size: self.deq_size,
            deq_filled: self.deq_filled,
            idv_score_init: self.idv_score_init
        }
    }
}

static mut POP: LinkedList<Individual> = LinkedList::new();
static mut GENES: LinkedList<Individual> = LinkedList::new();
static mut DATA_F: Vec<f32> = Vec::new();
static mut ANSWER: usize = 0;
static mut PENALTY: f32 = 0.0;

fn main() {
    let num_idv: usize = 10;
    let num_weight_type: usize = 10;
    let num_output_type: usize = 10;
    let num_space: usize = 100;
    let num_neuron: usize = 10;
    let num_axonbranch: usize = 10;
    let threshold: f32 = 10.0;
    let input_unit: usize = 1000;
    let reuptake: f32 = 0.99;
    let score_init: f32 = 50.0;
    let score_max: f32 = 500.0;
    let threshold_out: f32 = 50.0;
    let threshold_adult: usize = 20;
    let out_max: f32 = 300.0;
    let deq_size_init: usize = 20;
    
    let idv_score_init: f32 = 24.0;
    let bernoulli: Bernoulli = Bernoulli::new(0.03).unwrap();
    let penalty: f32 = 1.2;
    let idv_max: usize = 100;

    unsafe {
        PENALTY = penalty;
    }

    (0 .. num_idv).into_iter().for_each(|_| {
        let new_individual: Individual = init_individual (
            &num_weight_type,
            &num_output_type,
            &num_space,
            &num_neuron,
            &num_axonbranch,
            &threshold,
            &input_unit,
            &reuptake,
            &score_init,
            &score_max,
            &threshold_out,
            &idv_score_init,
            &threshold_adult,
            &out_max,
            &deq_size_init
        );
        unsafe {
            POP.push_back(new_individual.clone());
            GENES.push_back(new_individual);

        }
    });

    let mut rng = rand::thread_rng();
    let paths = fs::read_dir("C:/Users/qy94h/Downloads/dogs-vs-cats/train/train").unwrap();
    let mut list_paths: Vec<(PathBuf, usize)> = Vec::new();
    
    paths.for_each(|path| {
        let x0 = path.as_ref().unwrap().path().into_os_string().into_string().unwrap();
        let ans: usize = if x0.get(50 .. 53).unwrap() == "cat" {
            0
        }
        else if x0.get(50 .. 53).unwrap() == "dog" {
            1
        }
        else {
            panic!()
        };
            list_paths.push((path.unwrap().path(), ans));
    });

    let mut gen: usize = 0;
    unsafe {
        loop {

            //env. init

            list_paths.shuffle(&mut rng);

            list_paths.iter().for_each(|(path, ans)| {

                //action

                let data_u = fs::read(path).unwrap();
                DATA_F.clear();
                for u in data_u {
                    DATA_F.push(u as f32);
                }

                let mut handles = Vec::new();

                ANSWER = *ans;

                POP.iter_mut().for_each(|idv| {
                    let handle = thread::spawn(move || {
                        idv.action(&DATA_F, &PENALTY, &ANSWER);
                    });
                    handles.push(handle);
                });

                for handle in handles {
                    handle.join().unwrap();
                }

                //next gen

                if POP.len() >= idv_max {
                    let mut list_adult: Vec<f32> = Vec::new();
    
                    POP.iter().for_each(|idv| {
                        if idv.adult == true {
                            list_adult.push(idv.score);
                        }
                        else {
                            list_adult.push(1000.0);
                        }
                    });
    
                    (0 .. 2).for_each(|_| {
                        let dead: usize = argmin(&list_adult);
                        POP.remove(dead);
                        GENES.remove(dead);
                        list_adult.remove(dead);
                    });
                }

                let mut sum: f32 = 0.0;
                
                POP.iter().for_each(|idv| {
                    sum += idv.score;
                });

                let mut handles2 = Vec::new();

                POP.iter_mut().zip(GENES.iter_mut()).for_each(|(idv, gene)| {
                    if idv.adult == true && rng.gen_range(0.0 .. sum) < idv.score {
                        let handle = thread::spawn(move || {
                            let mut newborn = gene.clone();
                            newborn.mutation(&bernoulli);
                            newborn
                        });
                        handles2.push(handle);
                    }
                });

                for handle in handles2 {
                    let new = handle.join().unwrap();
                    POP.push_back(new.clone());
                    GENES.push_back(new);
                }

                let mut scores: Vec<f32> = Vec::new();
                let mut ages: Vec<usize> = Vec::new();
                
                POP.iter().for_each(|idv| {
                    scores.push(idv.score);
                    ages.push(idv.age);
                });
                let score_min = min(&scores);
                let score_max = max(&scores);
                let score_mean = mean_f(&scores);
    
                let age_max = max(&ages);
                let age_mean = mean_usize(&ages);
    
                gen += 1;
    
                println!("{}, score min = {:.2}, mean = {:.2}, max = {:.2}, age mean = {:.2}, max = {}", gen, score_min, score_mean, score_max, age_mean, age_max);
            });
        }
    }
}