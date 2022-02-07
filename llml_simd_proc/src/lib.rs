use proc_macro2::{Literal, TokenStream};
use quote::{quote, ToTokens};
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

// ARRAY GENERATOR
use syn::parse::Parse;
use syn::punctuated::Punctuated;
use syn::token::Comma;

struct ArrInput {
    expr: Expr,
    len: Lit
}

impl Parse for ArrInput {
    fn parse(input: parse::ParseStream) -> Result<Self> {
        let expr = input.parse::<Expr>()?;
        input.parse::<Token![;]>()?;
            
        Ok(ArrInput {
            expr,
            len: input.parse::<Lit>()?
        })
    }
}

#[proc_macro]
pub fn arr (input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as ArrInput);

    let len = match input.len {
        Lit::Int(x) => x.base10_parse::<usize>().unwrap(),
        _ => panic!("Only integers are valid as array lengths")
    };

    let expresions = (0..len).into_iter()
        .map(|i| match input.expr.clone() {
            Expr::Lit(lit) => lit.into_token_stream(),
            Expr::Closure(c) => {
                assert!(c.inputs.len() == 1 && matches!(&c.inputs[0], Ident), "Invalid expresion");
                parse_quote! { (#c)(#i) }
            },
            _ => panic!("Invalid array input")
        }).collect::<Punctuated<TokenStream, Comma>>();

    let output = quote! { [#expresions] };
    output.into()
}