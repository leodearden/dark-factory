/// Processor built on top of crate_a (≈ reify-compiler).
use train_a::BaseNode;

pub struct Processor {
    pub node: BaseNode,
}

impl Processor {
    pub fn new(value: u32) -> Self {
        Self { node: BaseNode::new(value) }
    }

    pub fn processed_value(&self) -> u32 {
        self.node.value * 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn processor_doubles_value() {
        let p = Processor::new(7);
        assert_eq!(p.processed_value(), 14);
    }
}
