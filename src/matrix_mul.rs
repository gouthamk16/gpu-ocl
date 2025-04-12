use ocl::{ProQue, Result}; // ocl is the Rust crate for opencl, ProQue is a helper that combines an openCL program, queue and context
use std::time::Instant; // Import for timing measurements

pub fn matmul() -> Result<()> {
    const N: usize = 32;
    const SIZE: usize = N * N;
    
    // Start overall timing
    let overall_start = Instant::now();

    // Defining an openCL kernel for 2D array multiplication
    let matmul_kernel = r#"
        __kernel void matmul(
            __global const float *a,
            __global const float *b,
            __global float *c,
            const int N
        ) {
            int row = get_global_id(0);
            int col = get_global_id(1);
            float sum = 0.0f;
            for (int k=0; k<N; ++k) {
                sum += a[row * N + k] * b[k * N + col];
            }
            c[row * N + col] = sum;
        }
    "#;

    // Setting up openCL,  computation pipeline
    let setup_start = Instant::now();
    let proque = ProQue::builder()
        .src(matmul_kernel) // Loads the kernel we defined earlier
        .dims([N, N]) // num of ||el threads
        .build()?; // Builds the openCL program
    let setup_time = setup_start.elapsed();
    println!("OpenCL setup time: {:?}", setup_time);

    // Creating the sample input data
    let a_data: Vec<f32> = (0..SIZE).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..SIZE).map(|i| i as f32).collect();

    // Creating buyffers for GPU memory
    let a_buffer = proque.create_buffer::<f32>()?;
    let b_buffer = proque.create_buffer::<f32>()?;
    let c_buffer = proque.create_buffer::<f32>()?;

    // Writing data to GPU buffers
    let write_start = Instant::now();
    a_buffer.cmd().write(&a_data).enq()?;
    b_buffer.cmd().write(&b_data).enq()?;
    let write_time = write_start.elapsed();
    println!("Buffer write time: {:?}", write_time);

    //Building and setting up the kernel
    let kernel_build_start = Instant::now();
    let kernel = proque.kernel_builder("matmul")
        .arg(&a_buffer)
        .arg(&b_buffer)
        .arg(&c_buffer)
        .arg(&(N as i32))
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
    let mut c_data: Vec<f32> = vec![0.0f32; SIZE];
    c_buffer.read(&mut c_data).enq()?;
    let read_time = read_start.elapsed();
    println!("Buffer read time: {:?}", read_time);

    // Verify output
    let expected = |row: usize, col: usize| -> f32 {
        let mut sum = 0.0f32;
        for k in 0..N {
            sum += a_data[row * N + k] * b_data[k * N + col];
        }
        sum
    };

    for row in 0..N {
        for col in 0..N {
            let idx = row * N + col;
            let expected_val = expected(row, col);
            assert!((c_data[idx] - expected_val).abs() < 1e-3);
        }
    }

    // Log total time
    println!("Matmul done.");
    let total_time = overall_start.elapsed();
    println!("Total execution time: {:?}", total_time);

    Ok(())
}