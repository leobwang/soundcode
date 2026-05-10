mod my_module;

use http::Request;
use my_module::response;

fn main() {
    let req = Request::builder().uri("/foo").body(()).unwrap();
    let resp = response(req).unwrap();
    println!("status = {}", resp.status());
}
