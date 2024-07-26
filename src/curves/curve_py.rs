

/// Convert the `nodes`of a `Curve` from a `HashMap` input form into the local data model.
/// Will upcast f64 values to a new ADOrder adding curve variable tags by id.
fn hashmap_into_nodes_timestamp(
    h: HashMap<NaiveDateTime, DualsOrF64>,
    ad: ADOrder,
    id: &str,
) -> NodesTimestamp {
    let vars: Vec<String> = get_variable_tags(id, h.keys().len());

    /// First convert to IndexMap and sort key order.
    let mut im: IndexMap<i64, DualsOrF64> = IndexMap::from_iter(h.into_iter().map(|(k,v)| (k.and_utc().timestamp(), v)));
    im.sort_keys();

    match ad {
        ADOrder::Zero => { NodesTimestamp::F64(IndexMap::from_iter(im.into_iter().map(|(k,v)| (k, f64::from(v))))) }
        ADOrder::One => { NodesTimestamp::Dual(IndexMap::from_iter(im.into_iter().enumerate().map(|(i,(k,v))| (k, Dual::from(set_order_with_conversion(v, ad, vec![vars[i].clone()])))))) }
        ADOrder::Two => { NodesTimestamp::Dual2(IndexMap::from_iter(im.into_iter().enumerate().map(|(i,(k,v))| (k, Dual2::from(set_order_with_conversion(v, ad, vec![vars[i].clone()])))))) }
    }
}