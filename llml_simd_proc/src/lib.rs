include!("assign.rs");
use quote::quote;
use syn::*;

#[proc_macro_attribute]
pub fn assign_targets (_input: proc_macro::TokenStream, alt: proc_macro::TokenStream) -> proc_macro::TokenStream {
    alt
}

#[proc_macro_attribute]
pub fn assign_rhs (_input: proc_macro::TokenStream, alt: proc_macro::TokenStream) -> proc_macro::TokenStream {
    alt
}

#[proc_macro_derive(Assign)]
pub fn assign_macro (input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let DeriveInput { 
        generics, 
        attrs, 
        vis: _, 
        ident, 
        data
    } = parse_macro_input!(input as DeriveInput);

    let targets = attrs.iter()
        .filter_map(|x| match x.parse_meta().unwrap() {
            Meta::List(list) => if list.path.is_ident("assign_targets") {
                return Some(list.nested.into_iter()
                    .map(|y| match y {
                        NestedMeta::Meta(meta) => meta.path().get_ident().unwrap().clone(),
                        _ => panic!("Unexpected error")
                    }))
            } else { None },
            _ => None
        })
        .flatten();

    let rhss = attrs.iter()
        .filter_map(|x| match x.parse_meta().unwrap() {
            Meta::List(list) => if list.path.is_ident("assign_rhs") {
                return Some(list.nested.into_iter()
                    .map(|y| match y {
                        NestedMeta::Meta(meta) => meta.path().get_ident().unwrap().clone(),
                        _ => panic!("Unexpected error")
                    }))
            } else { None },
            _ => None
        })
        .flatten()
        .collect::<Vec<Ident>>();

    let mut output = Vec::new();
    for target in targets {
        for rhs in rhss.iter() {
            output.push(assign_macro_impl(ident.clone(), generics.clone(), target.clone(), rhs.clone()));
        }
    }

    let output = quote! { #(#output)* };
    output.into()
}

fn assign_macro_impl (target: Ident, generics: Generics, original: Ident, rhs: Ident) -> proc_macro2::TokenStream {
    let original_name = format!("{original}");
    let original_fun = Ident::new(original_name.to_lowercase().as_str(), original.span());

    let assign_trait = Ident::new(&format!("{}Assign", original_name), original.span());
    let assign_fun = Ident::new(&format!("{}_assign", original_name.to_lowercase()), original.span());

    quote! {
        impl #assign_trait<#rhs> for #target #generics {
            fn #assign_fun (&mut self, rhs: #rhs) {
                *self = #original::<#rhs>::#original_fun(*self, rhs)
            } 
        } 
    }
}