/// Base type provided by crate_a (≈ reify-ir).
pub struct BaseNode {
    pub value: u32,
}

impl BaseNode {
    pub fn new(value: u32) -> Self {
        Self { value }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_node_roundtrip() {
        let n = BaseNode::new(42);
        assert_eq!(n.value, 42);
    }
}
