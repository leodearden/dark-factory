/// Evaluator built on top of crate_b (≈ reify-eval).
use train_b::Processor;

pub struct Evaluator {
    pub processor: Processor,
}

impl Evaluator {
    pub fn new(value: u32) -> Self {
        Self { processor: Processor::new(value) }
    }

    pub fn evaluated_value(&self) -> u32 {
        self.processor.processed_value() + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluator_increments_processed() {
        let e = Evaluator::new(3);
        // 3 * 2 + 1 = 7
        assert_eq!(e.evaluated_value(), 7);
    }
}
