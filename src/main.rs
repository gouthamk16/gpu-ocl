use ocl::{ProQue, Result}; // ocl is the Rsut crate for opencl, ProQue is a helper that combines an openCL program, queue and context
use std::time::{Instant, Duration}; // Import for timing measurements

fn main() -> Result<()> {
    // Start overall timing
    let overall_start = Instant::now();
    
    // Defining the openCL program -> adds two arrays and stores them in another, gets the index, thread based ||el execution
    let kernel_src = r#" 
        __kernel void add(
            __global const float* a,
            __global const float* b,
            __global float* c
        ) {
            int i = get_global_id(0);
            c[i] = a[i] + b[i]; 
        }
    "#;

    // Setting up openCL,  computation pipeline
    let setup_start = Instant::now();
    let proque = ProQue::builder()
        .src(kernel_src) // Loads the kernel we defined earlier
        .dims(1024) // num of ||el threads
        .build()?; // Builds the openCL program
    let setup_time = setup_start.elapsed();
    println!("OpenCL setup time: {:?}", setup_time);

    // Creating the sample input data
    let a_data = vec![1.0f32; 1024];
    let b_data = vec![2.0f32; 1024];

    // Creating buyffers for GPU memory
    let a_buffer = proque.create_buffer::<f32>()?;
    let b_buffer = proque.create_buffer::<f32>()?;
    let c_buffer = proque.create_buffer::<f32>()?;

    // Writing data to GPU
    let write_start = Instant::now();
    a_buffer.cmd().write(&a_data).enq()?;
    b_buffer.cmd().write(&b_data).enq()?;
    let write_time = write_start.elapsed();
    println!("Buffer write time: {:?}", write_time);

    //Building and setting up the kernel
    let kernel_build_start = Instant::now();
    let kernel = proque.kernel_builder("add")
        .arg(&a_buffer)
        .arg(&b_buffer)
        .arg(&c_buffer)
        .build()?;
    let kernel_build_time = kernel_build_start.elapsed();
    println!("Kernel build time: {:?}", kernel_build_time);

    // Execute the kernel
    let execution_start = Instant::now();
    unsafe { kernel.enq()?; }
    let execution_time = execution_start.elapsed();
    println!("Kernel execution time: {:?}", execution_time);

    // Read result back
    let read_start = Instant::now();
    let mut c_data: Vec<f32> = vec![0.0f32; 1024];
    c_buffer.cmd().read(&mut c_data).enq()?;
    let read_time = read_start.elapsed();
    println!("Buffer read time: {:?}", read_time);

    // Verify output
    for &c in &c_data {
        assert_eq!(c, 3.0f32);
    }

    // Log total time
    let overall_time = overall_start.elapsed();
    println!("Total execution time: {:?}", overall_time);

    Ok(())
}